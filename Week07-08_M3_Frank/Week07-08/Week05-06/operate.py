# teleoperate the robot, perform SLAM, planning, and waypoint following
# with burst driving, adaptive scans, continuous map alignment, and no-bump safeguards

import os
import sys
import time
import json
import re
import cv2
import math
import numpy as np
from collections import OrderedDict

# ---------- Utility & GUI ----------
sys.path.insert(0, "{}/util".format(os.getcwd()))
from util.pibot import PenguinPi
import util.DatasetHandler as dh
import util.measure as measure
import pygame
import shutil

# ---------- SLAM ----------
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco

# ---------- Detector (optional) ----------
from YOLO.detector import Detector

# ---------- Path Planner ----------
from path_planner import SafeLOSPlanner


# ===============================
# Navigation / Control Parameters
# ===============================
NAV_CONFIG = {
    # --- Waypoint navigation tolerances ---
    "heading_tolerance": math.radians(5.0),   # [rad] acceptable heading error
    "waypoint_tolerance": 0.10,               # [m]

    # --- Goal handling ---
    "goal_wait_seconds": 2.0,                 # [s] stop at each goal

    # --- Motion command magnitudes (normalized -1..+1) ---
    "fwd_speed": 0.6,
    "turn_speed": 0.6,

    # --- Path replanning ---
    "path_replan": True,
    "path_replan_trigger": 0.15,              # [m] tighter so we keep clear of obstacles

    # --- File inputs ---
    "slam_map_file": "slam.txt",
    "shopping_list_file": "shopping_list.txt",
    # Reference map for both obstacles and alignment (preferred)
    "reference_map_file": "slam_ref.json",    # JSON with aruco_* entries; falls back to slam.txt if absent

    # --- Adaptive Scan rhythm (drive-burst -> scan) ---
    # Baseline micro scan (< 1 s)
    "scan_period_s": 1.8,          # min seconds between scans
    "scan_min_dist_m": 0.25,       # also require this much forward progress
    "scan_total_s": 0.60,          # total time for micro scan
    "scan_phase_left_s": 0.18,     # left nudge
    "scan_phase_right_s": 0.18,    # right nudge
    "scan_phase_still_s": 0.24,    # still to let EKF add landmarks

    # Wide scan (used if several scans found few/no tags)
    "wide_scan_total_s": 1.60,
    "wide_scan_angles_deg": [45, -45, 90, -90, 135, -135],  # stepped yaw targets
    "wide_scan_dwell_s": 0.18,      # pause per angle to add tags

    # Full 360 scan (fallback if stuck/corner/edge)
    "full_scan_total_s": 2.8,       # continuous rotate in place
    "full_scan_turn_cmd": 1,        # +1 or -1

    # Escalation thresholds
    "no_tag_scan_escalate": 2,      # after N consecutive micro scans with no new tags -> wide
    "no_tag_wide_escalate": 1,      # after M wide scans with no new tags -> full 360

    # --- Scan placement rules ---
    "min_dist_to_wp_for_scan_m": 0.20,  # avoid scanning at a waypoint

    # --- Safety: brake/scan if a known obstacle is ahead and too close ---
    "forward_brake_dist_m": 0.22,       # stop if any known obstacle is within this
    "forward_brake_fov_deg": 35.0,      # +/- FOV as "ahead"

    # --- Bounds (keep within known area bbox with margin) ---
    "bounds_margin_m": 0.25,

    # --- Obstacle radii (conceptual; planner may treat as points, we still use for local checks) ---
    "object_radius_m": 0.10,
    "aruco_radius_m": 0.12
}

def ang_wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


class Operate:
    def __init__(self, args, TITLE_FONT, TEXT_FONT):
        self.args = args
        self.folder = 'pibot_dataset/'
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        else:
            shutil.rmtree(self.folder)
            os.makedirs(self.folder)

        # dataset / pibot
        if args.play_data:
            self.pibot = dh.DatasetPlayer("record")
        else:
            self.pibot = PenguinPi(args.ip, args.port)

        # SLAM
        self.ekf = self.init_ekf(args.calib_dir, args.ip)
        self.aruco_det = aruco.aruco_detector(self.ekf.robot, marker_length=0.07)

        if args.save_data:
            self.data = dh.DatasetWriter('record')
        else:
            self.data = None
        self.output = dh.OutputWriter('lab_output')

        self.command = {
            'motion': [0, 0],
            'inference': False,
            'output': False,
            'save_inference': False,
            'save_image': False
        }
        self.quit = False
        self.notification = 'SLAM initializing...'
        self.request_recover_robot = False
        self.ekf_on = True    # EKF always running unless paused manually
        self.image_id = 0

        # timers
        self.count_down = 300
        self.start_time = time.time()
        self.control_clock = time.time()

        # images
        self.img = np.zeros([240, 320, 3], dtype=np.uint8)
        self.aruco_img = np.zeros([240, 320, 3], dtype=np.uint8)
        self.detector_output = np.zeros([240, 320], dtype=np.uint8)
        if args.yolo_model == "":
            self.detector = None
            self.yolo_vis = cv2.imread('pics/8bit/detector_splash.png')
        else:
            self.detector = Detector(args.yolo_model)
            self.yolo_vis = np.ones((240, 320, 3)) * 100
        self.bg = pygame.image.load('pics/gui_mask.jpg')

        # --- GUI text surfaces (static captions) ---
        self.caption_surfaces = {
            "SLAM": TITLE_FONT.render("SLAM", False, (200, 200, 200)),
            "Detector": TITLE_FONT.render("Detector", False, (200, 200, 200)),
            "PiBot Cam": TITLE_FONT.render("PiBot Cam", False, (200, 200, 200))
        }

        # --- Notification caching ---
        self.last_notification = None
        self.notification_surface = None
        self.TEXT_FONT = TEXT_FONT

        # Navigation state & maps
        self.map_objects, self.markers_from_file = self._load_slam_map(NAV_CONFIG["slam_map_file"])
        self.reference_tags = self._load_reference_aruco_map(
            NAV_CONFIG["reference_map_file"]
        ) or self._extract_aruco_from_txt(self.map_objects)

        # bbox for bounds
        self.map_bbox = self._compute_bbox(self.map_objects, self.reference_tags)

        self.shopping_order = self._load_shopping_list(NAV_CONFIG["shopping_list_file"])
        self.remaining_targets = self._resolve_targets_to_positions(self.shopping_order, self.map_objects)

        self.curr_goal_name = None
        self.curr_goal_xy = None
        self.rx, self.ry = [], []
        self.wp_idx = 0
        self.last_plan_time = 0.0
        self.last_path = []
        self.waiting_until = None
        self.last_goal_name = None

        self.auto_mode = True

        # --- Scan state ---
        self.scan_active = False
        self.scan_mode = "micro"       # "micro" -> "wide" -> "full"
        self.scan_t0 = 0.0
        self.scan_end = 0.0
        self.scan_phase_end = []       # for stepped/wide angles
        self.scan_phase_idx = 0
        self.scans_without_tags = 0
        self.wide_scans_without_tags = 0
        self.last_scan_pose = (0.0, 0.0)
        self.last_scan_time = time.time()
        self.scan_initted = False

        # --- Alignment state ---
        self.map_aligned = False

    # ----------------- Wheel control -----------------
    def control(self):
        if self.args.play_data:
            lv, rv = self.pibot.set_velocity()
        else:
            lv, rv = self.pibot.set_velocity(self.command['motion'])

        if self.data is not None:
            self.data.write_keyboard(lv, rv)

        dt = time.time() - self.control_clock
        if self.args.ip == 'localhost':
            drive_meas = measure.Drive(lv, rv, dt)
        else:
            drive_meas = measure.Drive(lv, -rv, dt)
        self.control_clock = time.time()
        return drive_meas

    # ----------------- Camera -----------------
    def take_pic(self):
        self.img = self.pibot.get_image()
        if self.data is not None:
            self.data.write_image(self.img)

    # ----------------- SLAM -----------------
    def update_slam(self, drive_meas):
        lms, self.aruco_img = self.aruco_det.detect_marker_positions(self.img)
        # count how many unique tags detected this frame
        tags_seen = len({lm.tag for lm in lms}) if lms else 0

        if self.request_recover_robot:
            is_success = self.ekf.recover_from_pause(lms)
            if is_success:
                self.notification = 'Robot pose recovered'
                self.ekf_on = True
            else:
                self.notification = 'Recover failed'
                self.ekf_on = False
            self.request_recover_robot = False
        elif self.ekf_on:
            self.ekf.predict(drive_meas)
            # EKF only adds when still; scans ensure stillness moments
            self.ekf.add_landmarks(lms)
            self.ekf.update(lms)

            # Adaptive escalation bookkeeping
            if self._in_scan_window():
                if tags_seen == 0:
                    if self.scan_mode == "micro":
                        self.scans_without_tags += 1
                    elif self.scan_mode == "wide":
                        self.wide_scans_without_tags += 1
                else:
                    # reset counters if we saw something
                    self.scans_without_tags = 0
                    self.wide_scans_without_tags = 0

            # Align to reference map (preferred via EKF helper; else do it here)
            self._align_to_reference_if_possible()

    def _align_to_reference_if_possible(self):
        if not self.reference_tags:
            return
        aligned_now = False
        if hasattr(self.ekf, "merge_with_reference_map"):
            merged = self.ekf.merge_with_reference_map(self.reference_tags, min_matches=2)
            aligned_now = bool(merged)
        else:
            # Manual Umeyama alignment using EKF landmarks & taglist
            if not hasattr(self.ekf, "taglist") or self.ekf.number_landmarks() < 2:
                return
            # build matches
            pairs_ref = []
            pairs_live = []
            for i, tag in enumerate(self.ekf.taglist):
                if int(tag) in self.reference_tags:
                    pairs_live.append(self.ekf.markers[:, i:i+1])        # 2x1
                    xy_ref = np.array(self.reference_tags[int(tag)]).reshape(2,1)
                    pairs_ref.append(xy_ref)
            if len(pairs_ref) >= 2:
                from_pts = np.concatenate(pairs_live, axis=1)  # 2xN (live)
                to_pts   = np.concatenate(pairs_ref, axis=1)   # 2xN (ref)
                try:
                    R, t = EKF.umeyama(from_pts, to_pts)       # live -> ref
                    th_R = math.atan2(R[1,0], R[0,0])
                    # transform robot
                    x, y, th = self.ekf.robot.state.flatten()
                    p_ref = R @ np.array([[x],[y]]) + t
                    self.ekf.robot.state[0,0] = float(p_ref[0,0])
                    self.ekf.robot.state[1,0] = float(p_ref[1,0])
                    self.ekf.robot.state[2,0] = ang_wrap(th + th_R)
                    # transform landmarks
                    if self.ekf.number_landmarks() > 0:
                        self.ekf.markers = (R @ self.ekf.markers) + t
                    aligned_now = True
                except Exception:
                    pass

        if aligned_now:
            self.map_aligned = True
            self.notification = "Aligned to reference map ✅"

    # wheel and camera calibration for SLAM
    def init_ekf(self, datadir, ip):
        fileK = "{}intrinsic.txt".format(datadir)
        camera_matrix = np.loadtxt(fileK, delimiter=',')
        fileD = "{}distCoeffs.txt".format(datadir)
        dist_coeffs = np.loadtxt(fileD, delimiter=',')
        fileS = "{}scale.txt".format(datadir)
        scale = np.loadtxt(fileS, delimiter=',')
        if ip == 'localhost':
            scale /= 2
        fileB = "{}baseline.txt".format(datadir)
        baseline = np.loadtxt(fileB, delimiter=',')
        robot = Robot(baseline, scale, camera_matrix, dist_coeffs)
        return EKF(robot)

    # ----------------- Keyboard -----------------
    def update_keyboard(self):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.auto_mode = not self.auto_mode
                    self.notification = f"Auto mode {'resumed' if self.auto_mode else 'paused'}"
                elif event.key == pygame.K_p:  # toggle YOLO
                    self.command['inference'] = not self.command['inference']
                    self.notification = f"YOLO {'enabled' if self.command['inference'] else 'disabled'}"
                elif event.key == pygame.K_ESCAPE:
                    self.quit = True
            elif event.type == pygame.QUIT:
                self.quit = True
        if self.quit:
            pygame.quit()
            sys.exit()

    # ----------------- Main navigation orchestrator -----------------
    def run_navigation_step(self):
        if not self.auto_mode or not self.ekf_on:
            return

        # Scan controller preempts everything
        if self._run_scan_if_active():
            return

        # All done?
        if not self.remaining_targets and self.curr_goal_name is None:
            self.command['motion'] = [0, 0]
            self.notification = "All targets completed."
            return

        # Honor timed waits
        if self.waiting_until is not None:
            if time.time() < self.waiting_until:
                self.command['motion'] = [0, 0]
                return
            else:
                self.waiting_until = None

        # Pick a goal if none
        if self.curr_goal_name is None:
            self.curr_goal_name, self.curr_goal_xy = self.remaining_targets.popitem(last=False)
            self._plan_to_current_goal()
            # reset scan baselines
            self.last_scan_pose = self._xy()
            self.last_scan_time = time.time()
            self.scan_initted = True
            return

        # Replan if drifted
        if self._path_deviation_large():
            self._plan_to_current_goal()

        # Local safety: bounds + obstacle brake
        if not self._within_bounds_with_margin():
            # turn back inward and full scan to regain bearings
            self.notification = "↩️ Near/Outside bounds — turning in & scanning"
            self._start_full_scan()
            self.command['motion'] = [0, 0]
            return

        # Normal waypoint follower w/ scan scheduling and no-bump checks
        self._follow_waypoints()

    # ----------------- Planner -----------------
    def _plan_to_current_goal(self):
        sx, sy, _ = self._get_robot_pose()
        gx, gy = self.curr_goal_xy

        self.notification = (f"🗺️ Planning path from "
                             f"({sx:.2f}, {sy:.2f}) → ({gx:.2f}, {gy:.2f})")

        obstacles = self._compose_obstacles_for_planner()
        planner = SafeLOSPlanner(obstacles=obstacles, markers=self.markers_from_file)
        rx, ry = planner.plan(
            sx, sy, gx, gy,
            target_name=self.curr_goal_name,
            prev_target_name=self.last_goal_name
        )

        if not rx:
            self.rx, self.ry, self.wp_idx = [], [], 0
            self.notification = "❌ Planner failed: no valid path found"
            print(f"[Planner] ❌ No path found to {self.curr_goal_name}")
            return

        rx, ry = planner.smooth(rx, ry, gx, gy, target_name=self.curr_goal_name)
        self.rx, self.ry = rx, ry
        self.wp_idx = 1 if len(rx) > 1 else 0
        self.last_plan_time = time.time()
        self.last_path = list(zip(self.rx, self.ry))

        print(f"\n[Planner] ✅ Target: {self.curr_goal_name}")
        print("[Planner] Waypoints:")
        for i, (wx, wy) in enumerate(self.last_path):
            print(f"  {i}: ({wx:.2f}, {wy:.2f})")

        self.notification = f"✅ Path planned with {len(self.rx)} waypoints"
        self.last_goal_name = self.curr_goal_name

        # Optional visualization
        planner.plot_run(rx, ry, planner.obstacles, sx, sy, gx, gy, target_name=self.curr_goal_name)

    def _compose_obstacles_for_planner(self):
        obs = [(x, y, name) for name, (x, y) in self.map_objects.items()]
        # Add reference ArUco tags (treat markers as obstacles)
        for tid, (tx, ty) in self.reference_tags.items():
            obs.append((tx, ty, f"aruco_{tid}"))
        # Add EKF landmarks (live) — optional but helps avoid bumping
        if self.ekf.number_landmarks() > 0:
            for i, tag in enumerate(self.ekf.taglist):
                xy = self.ekf.markers[:, i]
                obs.append((float(xy[0]), float(xy[1]), f"aruco_live_{int(tag)}"))
        return obs

    # ----------------- Waypoint follower + scan scheduling + safety -----------------
    def _follow_waypoints(self):
        if not self.rx or self.wp_idx >= len(self.rx):
            self.command['motion'] = [0, 0]
            self.waiting_until = time.time() + NAV_CONFIG["goal_wait_seconds"]
            self.notification = "All waypoints completed"
            print(f"[Navigator] 🎯 Goal {self.curr_goal_name} reached. Pausing {NAV_CONFIG['goal_wait_seconds']:.1f}s.")
            self.curr_goal_name = None
            return

        sx, sy, th = self._get_robot_pose()
        gx, gy = self.rx[self.wp_idx], self.ry[self.wp_idx]

        # Emergency brake if obstacle ahead
        if not self._forward_clearance_ok(sx, sy, th):
            # Immediate micro/wide/full scan depending on escalation
            self._start_escalated_scan()
            self.command['motion'] = [0, 0]
            return

        # Opportunistic scan by cadence
        if self._should_start_scan(sx, sy, gx, gy):
            self._start_escalated_scan()
            self.command['motion'] = [0, 0]
            return

        # Normal stop–turn–go waypoint logic
        dx, dy = gx - sx, gy - sy
        d = math.hypot(dx, dy)

        if d < NAV_CONFIG["waypoint_tolerance"]:
            self.wp_idx += 1
            self.command['motion'] = [0, 0]

            if self.wp_idx >= len(self.rx):
                self.notification = f"✅ Target {self.curr_goal_name} reached"
                self.waiting_until = time.time() + NAV_CONFIG["goal_wait_seconds"]
                print(f"[Navigator] 🎯 Target {self.curr_goal_name} reached. Pausing {NAV_CONFIG['goal_wait_seconds']:.1f}s.")
                self.curr_goal_name = None
            else:
                self.notification = f"Waypoint {self.wp_idx} reached"
                self.waiting_until = time.time() + 0.5
                print(f"[Navigator] ⏸️ Waypoint {self.wp_idx} reached. Pausing 0.5s.")
            return

        desired_th = math.atan2(dy, dx)
        e_th = ang_wrap(desired_th - th)

        if abs(e_th) > NAV_CONFIG["heading_tolerance"]:
            turn_cmd = 1 if e_th > 0 else -1
            self.command['motion'] = [0, turn_cmd]
            self.notification = f"Rotating | error={math.degrees(e_th):.1f}°"
        else:
            self.command['motion'] = [1, 0]
            self.notification = f"Driving | dist={d:.2f}m"

    # ----------------- Scan scheduler / controller -----------------
    def _should_start_scan(self, sx, sy, gx, gy):
        # Avoid scans right at a waypoint
        if math.hypot(gx - sx, gy - sy) < NAV_CONFIG["min_dist_to_wp_for_scan_m"]:
            return False
        t_since = time.time() - self.last_scan_time
        d_since = dist((sx, sy), self.last_scan_pose)
        return (t_since >= NAV_CONFIG["scan_period_s"]) and (d_since >= NAV_CONFIG["scan_min_dist_m"])

    def _start_escalated_scan(self):
        # escalate based on recent lack of tags
        if self.scans_without_tags >= NAV_CONFIG["no_tag_scan_escalate"]:
            if self.wide_scans_without_tags >= NAV_CONFIG["no_tag_wide_escalate"]:
                self._start_full_scan()
                return
            self._start_wide_scan()
            return
        self._start_micro_scan()

    def _start_micro_scan(self):
        self.scan_mode = "micro"
        now = time.time()
        self.scan_active = True
        self.scan_t0 = now
        self.scan_end = now + NAV_CONFIG["scan_total_s"]
        self.scan_phase_end = [now + NAV_CONFIG["scan_phase_left_s"],
                               now + NAV_CONFIG["scan_phase_left_s"] + NAV_CONFIG["scan_phase_right_s"],
                               self.scan_end]  # still phase to end
        self.scan_phase_idx = 0
        self.notification = "🔎 Micro scan"

    def _start_wide_scan(self):
        self.scan_mode = "wide"
        now = time.time()
        self.scan_active = True
        self.scan_t0 = now
        self.scan_end = now + NAV_CONFIG["wide_scan_total_s"]
        n = len(NAV_CONFIG["wide_scan_angles_deg"])
        step = NAV_CONFIG["wide_scan_dwell_s"]
        self.scan_phase_end = [now + step*(i+1) for i in range(n)]
        # final still small pause
        self.scan_phase_end.append(self.scan_end)
        self.scan_phase_idx = 0
        self.notification = "🔭 Wide scan"

    def _start_full_scan(self):
        self.scan_mode = "full"
        now = time.time()
        self.scan_active = True
        self.scan_t0 = now
        self.scan_end = now + NAV_CONFIG["full_scan_total_s"]
        self.scan_phase_end = [self.scan_end]
        self.scan_phase_idx = 0
        self.notification = "🧭 360° scan"

    def _run_scan_if_active(self):
        if not self.scan_active:
            return False
        t = time.time()

        # End of scan
        if t >= self.scan_end:
            self.scan_active = False
            self.command['motion'] = [0, 0]
            sx, sy, _ = self._get_robot_pose()
            self.last_scan_pose = (sx, sy)
            self.last_scan_time = t
            # Replan after big scans to immediately exploit new alignment/tags
            if self.curr_goal_name is not None:
                self._plan_to_current_goal()
            return False

        # Drive pattern by mode
        if self.scan_mode == "micro":
            # left -> right -> still
            if t < self.scan_phase_end[0]:
                self.command['motion'] = [0, +1]
            elif t < self.scan_phase_end[1]:
                self.command['motion'] = [0, -1]
            else:
                self.command['motion'] = [0, 0]

        elif self.scan_mode == "wide":
            # step through angles by alternating left/right small bursts with dwells
            # simple implementation: alternate left/right turns every dwell slot, ending with still
            phase_count = len(self.scan_phase_end)
            if self.scan_phase_idx < phase_count - 1:
                # turning phase
                # alternate direction for variety
                turn_cmd = +1 if (self.scan_phase_idx % 2 == 0) else -1
                self.command['motion'] = [0, turn_cmd]
            else:
                # final still
                self.command['motion'] = [0, 0]
            if t >= self.scan_phase_end[self.scan_phase_idx]:
                self.scan_phase_idx += 1

        elif self.scan_mode == "full":
            # continuous rotate
            self.command['motion'] = [0, NAV_CONFIG["full_scan_turn_cmd"]]

        return True

    def _in_scan_window(self):
        return self.scan_active

    # ----------------- Safety: bounds & forward clearance -----------------
    def _within_bounds_with_margin(self):
        if self.map_bbox is None:
            return True
        (xmin, ymin, xmax, ymax) = self.map_bbox
        m = NAV_CONFIG["bounds_margin_m"]
        x, y, _ = self._get_robot_pose()
        return (xmin - m) <= x <= (xmax + m) and (ymin - m) <= y <= (ymax + m)

    def _forward_clearance_ok(self, sx, sy, th):
        fov = math.radians(NAV_CONFIG["forward_brake_fov_deg"])
        dmin = NAV_CONFIG["forward_brake_dist_m"]

        def ahead_and_close(ox, oy, radius):
            dx, dy = ox - sx, oy - sy
            d = math.hypot(dx, dy) - radius
            if d < 1e-6:
                return True
            bearing = ang_wrap(math.atan2(dy, dx) - th)
            return (abs(bearing) <= fov) and (d <= dmin)

        # Map objects
        r_obj = NAV_CONFIG["object_radius_m"]
        for _, (ox, oy) in self.map_objects.items():
            if ahead_and_close(ox, oy, r_obj):
                self.notification = "⛔ Obstacle ahead — scanning/avoid"
                return False

        # Reference ArUco tags
        r_tag = NAV_CONFIG["aruco_radius_m"]
        for _, (tx, ty) in self.reference_tags.items():
            if ahead_and_close(tx, ty, r_tag):
                self.notification = "⛔ Marker ahead — scanning/avoid"
                return False

        # Live EKF landmarks (optional)
        if self.ekf.number_landmarks() > 0:
            for i in range(self.ekf.number_landmarks()):
                lx = float(self.ekf.markers[0, i])
                ly = float(self.ekf.markers[1, i])
                if ahead_and_close(lx, ly, r_tag):
                    self.notification = "⛔ Live marker ahead — scanning/avoid"
                    return False

        return True

    # ----------------- Helpers -----------------
    def _path_deviation_large(self):
        if not NAV_CONFIG["path_replan"]:
            return False
        if not self.last_path:
            return False
        px, py, _ = self._get_robot_pose()
        dmin = min(dist((px, py), w) for w in self.last_path)
        if dmin > NAV_CONFIG["path_replan_trigger"]:
            self.notification = (f"⚠️ Replanning: drift {dmin:.2f}m > "
                                 f"{NAV_CONFIG['path_replan_trigger']}m")
            return True
        return False

    def _get_robot_pose(self):
        s = self.ekf.robot.state.reshape(-1)
        return float(s[0]), float(s[1]), float(s[2])

    def _xy(self):
        x, y, _ = self._get_robot_pose()
        return (x, y)

    def _load_slam_map(self, fname):
        if not os.path.exists(fname):
            return OrderedDict(), []
        with open(fname, "r") as f:
            obj = json.load(f)
        items = OrderedDict()
        markers = []
        for k, v in obj.items():
            x, y = float(v.get("x", 0)), float(v.get("y", 0))
            items[k] = (x, y)
            if "aruco" in k.lower():
                markers.append((x, y))
        return items, markers

    def _load_reference_aruco_map(self, fname):
        if not os.path.exists(fname):
            return {}
        with open(fname, "r") as f:
            obj = json.load(f)
        ref = {}
        for k, v in obj.items():
            if "aruco" in k.lower():
                m = re.search(r'(\d+)', k)
                if not m:
                    continue
                tag_id = int(m.group(1))
                x, y = float(v.get("x", 0)), float(v.get("y", 0))
                ref[tag_id] = (x, y)
        return ref

    def _extract_aruco_from_txt(self, items):
        # fallback: extract "aruco_*" keys from slam.txt-style dict
        ref = {}
        for k, (x, y) in items.items():
            if "aruco" in k.lower():
                m = re.search(r'(\d+)', k)
                if m:
                    ref[int(m.group(1))] = (x, y)
        return ref

    def _compute_bbox(self, objects_dict, ref_tags):
        pts = []
        for _, (x, y) in objects_dict.items():
            pts.append((x, y))
        for _, (x, y) in ref_tags.items():
            pts.append((x, y))
        if not pts:
            return None
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return (min(xs), min(ys), max(xs), max(ys))

    def _load_shopping_list(self, fname):
        if not os.path.exists(fname):
            return []
        with open(fname) as f:
            return [line.strip().lower() for line in f if line.strip()]

    def _resolve_targets_to_positions(self, names, map_objects):
        resolved = OrderedDict()
        for desired in names:
            for k, xy in map_objects.items():
                if k.lower().startswith(desired) and k not in resolved:
                    resolved[k] = xy
                    break
        return resolved

    # ----------------- GUI -----------------
    def draw(self, canvas):
        canvas.blit(self.bg, (0, 0))
        text_colour = (220, 220, 220)
        v_pad = 40
        h_pad = 20

        # --- SLAM outputs ---
        ekf_view = self.ekf.draw_slam_state(res=(320, 480 + v_pad),
                                            not_pause=self.ekf_on)
        canvas.blit(ekf_view, (2 * h_pad + 320, v_pad))
        robot_view = cv2.resize(self.aruco_img, (320, 240))
        self.draw_pygame_window(canvas, robot_view, position=(h_pad, v_pad))

        # --- Detector outputs ---
        detector_view = cv2.resize(self.yolo_vis, (320, 240), cv2.INTER_NEAREST)
        self.draw_pygame_window(canvas, detector_view, position=(h_pad, 240 + 2 * v_pad))

        # --- Static captions (pre-rendered) ---
        canvas.blit(self.caption_surfaces["SLAM"], (2 * h_pad + 320, v_pad - 25))
        canvas.blit(self.caption_surfaces["Detector"], (h_pad, 240 + 2 * v_pad - 25))
        canvas.blit(self.caption_surfaces["PiBot Cam"], (h_pad, v_pad - 25))

        # --- Notification text (re-render only if changed) ---
        if self.notification != self.last_notification:
            self.notification_surface = self.TEXT_FONT.render(self.notification, False, text_colour)
            self.last_notification = self.notification
        if self.notification_surface is not None:
            canvas.blit(self.notification_surface, (h_pad + 10, 596))

        return canvas

    @staticmethod
    def draw_pygame_window(canvas, cv2_img, position):
        cv2_img = np.rot90(cv2_img)
        view = pygame.surfarray.make_surface(cv2_img)
        view = pygame.transform.flip(view, True, False)
        canvas.blit(view, position)


# ===============================
# Program entry
# ===============================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default='192.168.50.1')
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--save_data", action='store_true')
    parser.add_argument("--play_data", action='store_true')
    parser.add_argument("--yolo_model", default='YOLO/model/best.pt')
    args, _ = parser.parse_known_args()

    pygame.font.init()
    TITLE_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 35)
    TEXT_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 40)

    width, height = 700, 660
    canvas = pygame.display.set_mode((width, height))
    pygame.display.set_caption('ECE4078 2025 Lab')
    pygame.display.set_icon(pygame.image.load('pics/8bit/pibot5.png'))
    canvas.fill((0, 0, 0))

    operate = Operate(args, TITLE_FONT, TEXT_FONT)

    while True:
        operate.update_keyboard()
        operate.take_pic()
        drive_meas = operate.control()
        operate.update_slam(drive_meas)
        operate.run_navigation_step()
        operate.draw(canvas)
        pygame.display.update()
