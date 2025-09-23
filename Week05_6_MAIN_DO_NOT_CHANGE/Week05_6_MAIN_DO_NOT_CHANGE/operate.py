# teleoperate the robot, perform SLAM, planning, and waypoint following

import os
import sys
import time
import json
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
from slam.ekf_markers import EKF
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
    "heading_tolerance": math.radians(2),   # [rad] acceptable heading error (~5°) before driving forward
    "waypoint_tolerance": 0.02,               # [m] distance tolerance (~10 cm) to consider a waypoint reached

    # --- Goal handling ---
    "goal_wait_seconds": 2.5,                 # [s] how long to stop at each goal before continuing

    # --- Motion command magnitudes (normalized: -1.0 to +1.0) ---
    "fwd_speed": 0.3,                         # forward driving speed
    "turn_speed": 0.3,                        # turning speed

    # --- Path replanning (safety mechanism) ---
    "path_replan": False,                     # enable/disable automatic replanning when drift is large
    "path_replan_trigger": 0.30,              # [m] trigger threshold: replan if robot deviates this far from path

    # --- File inputs ---
    "slam_map_file": "slam.txt",
    "shopping_list_file": "shopping_list.txt"
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

        # Navigation state
        self.map_objects, self.markers = self._load_slam_map(NAV_CONFIG["slam_map_file"])

        # === FIXED LANDMARKS ===
        # Load true ArUco marker positions into EKF (slam.txt defines them)
        marker_dict = {i: (x, y) for i, (x, y) in enumerate(self.markers)}
        self.ekf.set_fixed_landmarks(marker_dict)

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
            # add_landmarks does nothing now (landmarks fixed)
            self.ekf.add_landmarks(lms)
            self.ekf.update(lms)

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
                # pause / resume auto mode
                if event.key == pygame.K_SPACE:
                    self.auto_mode = not self.auto_mode
                    self.notification = f"Auto mode {'resumed' if self.auto_mode else 'paused'}"
                # toggle YOLO
                elif event.key == pygame.K_p:
                    self.command['inference'] = not self.command['inference']
                    self.notification = f"YOLO {'enabled' if self.command['inference'] else 'disabled'}"
                # quit
                elif event.key == pygame.K_ESCAPE:
                    self.quit = True
            elif event.type == pygame.QUIT:
                self.quit = True
        if self.quit:
            pygame.quit()
            sys.exit()

    # ----------------- Navigation main loop -----------------
    def run_navigation_step(self):
        if not self.auto_mode or not self.ekf_on:
            return
        if not self.remaining_targets and self.curr_goal_name is None:
            self.command['motion'] = [0, 0]
            self.notification = "All targets completed."
            return
        if self.waiting_until is not None:
            if time.time() < self.waiting_until:
                self.command['motion'] = [0, 0]
                return
            else:
                self.waiting_until = None
        if self.curr_goal_name is None:
            self.curr_goal_name, self.curr_goal_xy = self.remaining_targets.popitem(last=False)
            self._plan_to_current_goal()
            return
        if self._path_deviation_large():
            self._plan_to_current_goal()
        self._follow_waypoints()

    def _plan_to_current_goal(self):
        sx, sy, _ = self._get_robot_pose()
        gx, gy = self.curr_goal_xy

        self.notification = (f"🗺️ Planning path from "
                             f"({sx:.2f}, {sy:.2f}) → ({gx:.2f}, {gy:.2f})")

        planner = SafeLOSPlanner(
            obstacles=[(x, y, name) for name, (x, y) in self.map_objects.items()],
            markers=self.markers
        )
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

        # === TERMINAL OUTPUT ===
        print(f"\n[Planner] ✅ Target: {self.curr_goal_name}")
        print("[Planner] Waypoints:")
        for i, (wx, wy) in enumerate(self.last_path):
            print(f"  {i}: ({wx:.2f}, {wy:.2f})")

        self.notification = f"✅ Safe-LOS path planned with {len(self.rx)} waypoints"
        self.last_goal_name = self.curr_goal_name

        # === SHOW PLOT ===
        planner.plot_run(rx, ry, planner.obstacles, sx, sy, gx, gy, target_name=self.curr_goal_name)

    # ----------------- Stop–Turn–Go Waypoint Follower -----------------
    def _follow_waypoints(self):
        if not self.rx or self.wp_idx >= len(self.rx):
            # No waypoints left
            self.command['motion'] = [0, 0]
            self.waiting_until = time.time() + 2.5
            self.notification = "All waypoints completed"
            print(f"[Navigator] 🎯 Goal {self.curr_goal_name} reached. Pausing 2.5s.")
            if self.curr_goal_name is not None:
                sx, sy, th = self._get_robot_pose()
                print(f"[Navigator] Final Pose: x={sx:.2f}, y={sy:.2f}, θ={math.degrees(th):.1f}°")
            self.curr_goal_name = None
            return

        # --- Current pose & goal ---
        sx, sy, th = self._get_robot_pose()
        gx, gy = self.rx[self.wp_idx], self.ry[self.wp_idx]

        dx, dy = gx - sx, gy - sy
        d = math.hypot(dx, dy)
        desired_th = math.atan2(dy, dx)
        e_th = ang_wrap(desired_th - th)

        # --- Check if waypoint reached ---
        if d < NAV_CONFIG["waypoint_tolerance"]:
            self.wp_idx += 1
            self.command['motion'] = [0, 0]

            # Print robot pose at waypoint
            print(f"[Navigator] 📍 Waypoint reached: ({gx:.2f}, {gy:.2f})")
            print(f"[Navigator] Robot Pose: x={sx:.2f}, y={sy:.2f}, θ={math.degrees(th):.1f}°")

            if self.wp_idx >= len(self.rx):
                self.notification = f"✅ Target {self.curr_goal_name} reached"
                self.waiting_until = time.time() + 2.5
                print(f"[Navigator] 🎯 Target {self.curr_goal_name} reached. Pausing 2.5s.")
                self.curr_goal_name = None
            else:
                self.notification = f"Waypoint {self.wp_idx} reached"
                self.waiting_until = time.time() + 1
                print(f"[Navigator] ⏸️ Waypoint {self.wp_idx} reached. Pausing 0.5s.")
            return

        # --- Turn first if misaligned ---
        heading_thresh = NAV_CONFIG["heading_tolerance"]
        if abs(e_th) > heading_thresh:
            # Rotate until aligned
            turn_cmd = 1 if e_th > 0 else -1
            self.command['motion'] = [0, turn_cmd]
            self.notification = f"Rotating | error={math.degrees(e_th):.1f}°"
        else:
            # Drive straight forward (no turning at the same time)
            self.command['motion'] = [1, 0]
            self.notification = f"Driving straight | dist={d:.2f}m"

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
