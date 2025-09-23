import numpy as np
import cv2
import math
import pygame
from mapping_utils import MappingUtils


class EKF:
    """
    EKF for robot localization using known fixed ArUco landmarks.
    State order: [x; y; theta; l1x; l1y; ...; lnx; lny]

    - Landmarks are fixed: loaded from slam.txt (true positions)
    - Update step only corrects robot pose
    """

    # ===================== TUNABLES =====================
    BASE_MEAS_STD_M   = 0.03     # ~2 cm baseline
    GATE_CHI2         = 7.5      # gating threshold (≈97.5%)
    RANGE_INFLATE     = 0.7      # gentle inflation with distance
    Q_STAB            = 2e-5     # stabilizer on process noise

    # Range management
    MAX_RANGE_M   = 1.6
    FAR_REJECT    = False
    FAR_EXTRA_STD = 0.15

    # Optional alignment (for evaluation only)
    OPTIONAL_ALIGNMENT = False
    ALIGN_ROT_RAD   = -1.7354400353556125
    ALIGN_TX        = 0.2689604391944955
    ALIGN_TY        = 0.11240062133992615
    # ====================================================

    @staticmethod
    def _wrap_angle(a):
        return (a + np.pi) % (2*np.pi) - np.pi

    def __init__(self, robot):
        self.robot = robot
        self.markers = np.zeros((2,0))   # fixed landmark positions (global)
        self.taglist = []
        self.anchor_tag = None

        self.P = np.zeros((3,3))         # only robot covariance
        self.robot_init_state = None

        # UI assets
        self.lm_pics = []
        for i in range(1, 10+1):
            self.lm_pics.append(pygame.image.load(f'./pics/8bit/lm_{i}.png'))
        self.lm_pics.append(pygame.image.load(f'./pics/8bit/lm_unknown.png'))
        self.pibot_pic = pygame.image.load(f'./pics/8bit/pibot_top.png')

        # Optional alignment
        if self.OPTIONAL_ALIGNMENT:
            c, s = np.cos(self.ALIGN_ROT_RAD), np.sin(self.ALIGN_ROT_RAD)
            self.eval_R = np.array([[c, -s],[s, c]])
            self.eval_t = np.array([[self.ALIGN_TX],[self.ALIGN_TY]])
        else:
            self.eval_R = None
            self.eval_t = None

    # ---------------- Landmark management ----------------
    def set_fixed_landmarks(self, markers_dict):
        """
        Load known landmarks (aruco markers) from slam.txt into EKF.
        markers_dict: {tag_id: (x, y), ...} in global coordinates
        """
        self.taglist = list(markers_dict.keys())
        self.markers = np.array([[x for (x, y) in markers_dict.values()],
                                 [y for (x, y) in markers_dict.values()]])
        # Expand covariance matrix but keep landmark block fixed (=0)
        nL = len(self.taglist)
        self.P = np.block([
            [self.P, np.zeros((3, 2*nL))],
            [np.zeros((2*nL, 3)), np.zeros((2*nL, 2*nL))]
        ])

    def add_landmarks(self, measurements):
        # Overridden: no new landmarks added, since they are fixed
        return

    # ---------------- State helpers ----------------
    def number_landmarks(self):
        return int(self.markers.shape[1])

    def get_state_vector(self):
        return np.concatenate((self.robot.state,
                               np.reshape(self.markers, (-1,1), order='F')), axis=0)

    def set_state_vector(self, state):
        self.robot.state = state[0:3,:]
        self.markers = np.reshape(state[3:,:], (2,-1), order='F')

    def save_map(self, fname="slam_map.txt"):
        if self.number_landmarks() > 0:
            utils = MappingUtils(self.markers, self.P[3:,3:], self.taglist)
            utils.save(fname)

    # ---------------- Prediction ----------------
    def predict(self, drive_meas):
        lv, rv, dt = drive_meas.left_speed, drive_meas.right_speed, drive_meas.dt
        if abs(lv) < 1e-6 and abs(rv) < 1e-6:
            return  

        F = self.state_transition(drive_meas)
        self.robot.drive(drive_meas)
        Q = self.predict_covariance(drive_meas)
        self.P = F @ self.P @ F.T + Q

        x = self.get_state_vector()
        x[2,0] = self._wrap_angle(x[2,0])
        self.set_state_vector(x)

    def state_transition(self, raw_drive_meas):
        n = self.number_landmarks()*2 + 3
        F = np.eye(n)
        F[0:3,0:3] = self.robot.derivative_drive(raw_drive_meas)
        return F

    def predict_covariance(self, raw_drive_meas):
        n = self.number_landmarks()*2 + 3
        Q = np.zeros((n,n))
        Q[0:3,0:3] = self.robot.covariance_drive(raw_drive_meas) + self.Q_STAB*np.eye(3)
        return Q

    # ---------------- Update ----------------
    def _adaptive_R(self, z_i, d2=None):
        s = self.BASE_MEAS_STD_M
        Rb = np.diag([s*s, s*s])
        r = float(np.linalg.norm(z_i))
        if r < 0.5:
            R = Rb
        elif r <= self.MAX_RANGE_M:
            scale_r = 1.0 + self.RANGE_INFLATE * ((r-0.5)/0.9)
            scale_res = 1.0 if d2 is None else min(3.0, 0.5 + d2/self.GATE_CHI2)
            R = (scale_r * scale_res) * Rb
        else:
            R = Rb + np.diag([self.FAR_EXTRA_STD**2, self.FAR_EXTRA_STD**2])
        return R

    def _gate_one(self, z_i, zhat_i, H_i, R_i):
        S_i = H_i @ self.P @ H_i.T + R_i
        v_i = z_i - zhat_i
        d2  = float(v_i.T @ np.linalg.solve(S_i, v_i))
        return (d2 <= self.GATE_CHI2), d2

    def update(self, measurements):
        if not measurements or self.number_landmarks() == 0:
            return

        tags = [lm.tag for lm in measurements if lm.tag in self.taglist]
        if not tags:
            return

        idx_list = [self.taglist.index(tag) for tag in tags]
        z = np.concatenate([lm.position.reshape(-1,1) for lm in measurements if lm.tag in self.taglist], axis=0)
        z_hat = self.robot.measure(self.markers, idx_list).reshape((-1,1), order="F")
        H     = self.robot.derivative_measure(self.markers, idx_list)

        x = self.get_state_vector()

        kept_rows, R_adapt = [], np.zeros((2*len(idx_list),2*len(idx_list)))
        for i, tag in enumerate(tags):
            rr = slice(2*i, 2*i+2)
            z_i, zhat_i, H_i = z[rr, :], z_hat[rr, :], H[rr, :]
            R_i = np.diag([self.BASE_MEAS_STD_M**2]*2)
            keep, d2 = self._gate_one(z_i, zhat_i, H_i, R_i)
            if not keep:
                continue
            kept_rows.extend([2*i, 2*i+1])
            R_adapt[rr, rr] = self._adaptive_R(z_i, d2=d2)

        if not kept_rows:
            self.set_state_vector(x)
            return

        z_used    = z[kept_rows, :]
        zhat_used = z_hat[kept_rows, :]
        H_used    = H[kept_rows, :]
        R_used    = R_adapt[np.ix_(kept_rows, kept_rows)]

        S = H_used @ self.P @ H_used.T + R_used
        K = (self.P @ H_used.T) @ np.linalg.inv(S)

        innovation = z_used - zhat_used
        x = x + K @ innovation
        x[2,0] = self._wrap_angle(x[2,0])

        I = np.eye(self.P.shape[0])
        self.P = (I - K @ H_used) @ self.P @ (I - K @ H_used).T + K @ R_used @ K.T

        self.set_state_vector(x)

    # ---------------- Utilities & Drawing ----------------
    @staticmethod
    def umeyama(from_points, to_points):
        assert len(from_points.shape) == 2
        assert from_points.shape == to_points.shape
        N = from_points.shape[1]; m = 2
        mean_from = from_points.mean(axis=1).reshape((2,1))
        mean_to   = to_points.mean(axis=1).reshape((2,1))
        delta_from = from_points - mean_from
        delta_to   = to_points - mean_to
        cov_matrix = delta_to @ delta_from.T / N
        U, d, V_t = np.linalg.svd(cov_matrix, full_matrices=True)
        cov_rank = np.linalg.matrix_rank(cov_matrix)
        S = np.eye(m)
        if cov_rank >= m - 1 and np.linalg.det(cov_matrix) < 0:
            S[m-1, m-1] = -1
        elif cov_rank < m-1:
            raise ValueError("colinearity in covariance matrix")
        R = U.dot(S).dot(V_t)
        t = mean_to - R.dot(mean_from)
        return R, t

    @staticmethod
    def to_im_coor(xy, res, m2pixel):
        w, h = res
        x, y = xy
        x_im = int(-x*m2pixel+w/2.0)
        y_im = int(y*m2pixel+h/2.0)
        return (x_im, y_im)

    def draw_slam_state(self, res=(320, 500), not_pause=True):
        m2pixel = 100
        bg_rgb = np.array([213, 213, 213]).reshape(1, 1, 3) if not_pause else np.array([120, 120, 120]).reshape(1, 1, 3)
        canvas = np.ones((res[1], res[0], 3))*bg_rgb.astype(np.uint8)

        lms_xy = self.markers[:2, :]
        robot_xy = self.robot.state[:2, 0].reshape((2, 1))
        lms_xy = lms_xy - robot_xy
        robot_theta = self.robot.state[2,0]
        start_point_uv = self.to_im_coor((0, 0), res, m2pixel)

        # robot covariance ellipse
        p_robot = self.P[0:2,0:2]
        axes_len, angle = self.make_ellipse(p_robot)
        canvas = cv2.ellipse(canvas, start_point_uv,
                             (int(axes_len[0]*m2pixel), int(axes_len[1]*m2pixel)),
                             angle, 0, 360, (0, 30, 56), 1)

        # landmark ellipses (fixed -> zero cov, but keep rendering)
        if self.number_landmarks() > 0:
            for i in range(len(self.markers[0,:])):
                xy = (lms_xy[0, i], lms_xy[1, i])
                coor_ = self.to_im_coor(xy, res, m2pixel)
                Plmi = self.P[3+2*i:3+2*(i+1),3+2*i:3+2*(i+1)]
                axes_len, a = self.make_ellipse(Plmi)
                canvas = cv2.ellipse(canvas, coor_,
                                     (int(axes_len[0]*m2pixel), int(axes_len[1]*m2pixel)),
                                     a, 0, 360, (244, 69, 96), 1)

        surface = pygame.surfarray.make_surface(np.rot90(canvas))
        surface = pygame.transform.flip(surface, True, False)
        surface.blit(self.rot_center(self.pibot_pic, robot_theta*57.3),
                     (start_point_uv[0]-15, start_point_uv[1]-15))
        if self.number_landmarks() > 0:
            for i in range(len(self.markers[0,:])):
                xy = (lms_xy[0, i], lms_xy[1, i])
                coor_ = self.to_im_coor(xy, res, m2pixel)
                try:
                    surface.blit(self.lm_pics[self.taglist[i]-1], (coor_[0]-5, coor_[1]-5))
                except IndexError:
                    surface.blit(self.lm_pics[-1], (coor_[0]-5, coor_[1]-5))
        return surface

    @staticmethod
    def rot_center(image, angle):
        orig_rect = image.get_rect()
        rot_image = pygame.transform.rotate(image, angle)
        rot_rect = orig_rect.copy()
        rot_rect.center = rot_image.get_rect().center
        rot_image = rot_image.subsurface(rot_rect).copy()
        return rot_image

    @staticmethod
    def make_ellipse(P):
        if P.shape != (2,2):
            return (0,0), 0
        e_vals, e_vecs = np.linalg.eig(P)
        idx = e_vals.argsort()[::-1]
        e_vals = e_vals[idx]
        e_vecs = e_vecs[:, idx]
        alpha = np.sqrt(4.605)  # 90% confidence
        axes_len = np.sqrt(np.maximum(e_vals, 1e-12))*alpha
        angle = np.degrees(np.arctan2(e_vecs[1,0], e_vecs[0,0]))
        return (axes_len[0], axes_len[1]), angle
