# yolo_obs_logger.py
# 在运行时将 YOLO 检测到的（不在shopping_list里的）近距离水果/蔬菜写入 slam.txt，
# 并触发重规划（replan.flag）。可选触发回原点（home.flag）。

import os
import json
import time
import math
from types import SimpleNamespace

import numpy as np
import cv2

# --- 你的项目内模块 ---
from YOLO.detector import Detector
from util.pibot import PenguinPi
import pygame

# 直接复用 operate.Operate._get_robot_pose()
import operate as op


# =========================
# 固定默认配置（无需 argparse）
# =========================
INTRINSIC_PATH     = "calibration/param/intrinsic.txt"
SLAM_PATH          = "slam.txt"
SHOPPING_LIST_PATH = "shopping_list.txt"
YOLO_MODEL_PATH    = "YOLO/model/best.pt"

ROBOT_IP           = "192.168.50.1"
ROBOT_PORT         = 8080

LOOP               = True      # 持续循环
PERIOD_SEC         = 0.20      # 每次检测间隔
GO_HOME_ON_BLOCK   = False     # 触发近障时是否先回原点(0,0)，需要 operate.py 的旗标逻辑

# 触发与合并参数
TRIGGER_DIST       = 0.30      # [m] 30cm 内写入 slam.txt 并触发重规划
MERGE_THRESH       = 0.15      # [m] 与已有动态障碍的合并半径

# 相机与目标尺寸（用“高”估深度）
IMAGE_WIDTH        = 320
DIMS = {
    'orange':[0.0769,0.0747,0.071], 'lemon':[0.0502,0.0664,0.052],
    'pear':[0.07,0.078,0.0835],     'tomato':[0.0706,0.0688,0.062],
    'capsicum':[0.075,0.075,0.0935],'potato':[0.0678,0.0947,0.054],
    'pumpkin':[0.088,0.082,0.071],  'garlic':[0.0653,0.0653,0.0725],
}


# ==============
# 工具函数
# ==============
def load_fx(intrinsic_path):
    K = np.loadtxt(intrinsic_path, delimiter=',')
    return float(K[0][0])

def load_shopping_list(path):
    if not os.path.exists(path): return set()
    with open(path, 'r', encoding='utf-8') as f:
        return set([ln.strip().lower() for ln in f if ln.strip()])

def pinhole_distance(fx, true_h, pix_h):
    return fx * true_h / max(1e-6, float(pix_h))

def img_shift_angle(fx, x_center_px, img_w):
    x_shift = (img_w / 2.0) - float(x_center_px)
    return math.atan2(x_shift, fx)   # 左正右负（与像素坐标配合）

def body_to_world(rel_x, rel_y, robot_pose):
    x, y, th = float(robot_pose[0]), float(robot_pose[1]), float(robot_pose[2])
    dx_w = rel_x * math.cos(th) - rel_y * math.sin(th)
    dy_w = rel_x * math.sin(th) + rel_y * math.cos(th)
    return x + dx_w, y + dy_w

def load_slam(slam_path):
    if not os.path.exists(slam_path): return {}
    with open(slam_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_slam_atomic(slam_path, slam_obj):
    tmp = slam_path + ".tmp"
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(slam_obj, f, indent=2)
    os.replace(tmp, slam_path)

def upsert_obstacle(slam_obj, label, ox, oy, merge_thresh=MERGE_THRESH):
    """
    合并/新增动态障碍，键名 obs_<label>_<idx>
    """
    nearest_key, nearest_d = None, float('inf')
    for k, v in slam_obj.items():
        if not k.startswith(f"obs_{label}_"):
            continue
        try:
            d = math.hypot(ox - float(v.get('x', 0)), oy - float(v.get('y', 0)))
        except Exception:
            continue
        if d < nearest_d:
            nearest_d, nearest_key = d, k

    if nearest_key is not None and nearest_d <= merge_thresh:
        # 简单均值抑抖
        sx = (float(slam_obj[nearest_key]['x']) + ox) * 0.5
        sy = (float(slam_obj[nearest_key]['y']) + oy) * 0.5
        slam_obj[nearest_key]['x'] = sx
        slam_obj[nearest_key]['y'] = sy
        return nearest_key

    idx = sum(1 for k in slam_obj.keys() if k.startswith(f"obs_{label}_"))
    key = f"obs_{label}_{idx}"
    slam_obj[key] = {"x": float(ox), "y": float(oy)}
    return key


# ============================
# 主流程：初始化与循环检测
# ============================
def main():
    # 1) 读取相机内参、购物清单
    fx = load_fx(INTRINSIC_PATH)
    shopping_set = load_shopping_list(SHOPPING_LIST_PATH)

    # 2) 初始化 pygame 字体（Operate 需要字体对象传入）
    pygame.font.init()
    try:
        TITLE_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 35)
        TEXT_FONT  = pygame.font.Font('pics/8-BitMadness.ttf', 40)
    except Exception:
        # 如果字体文件不存在，就用默认字体
        TITLE_FONT = pygame.font.SysFont(None, 35)
        TEXT_FONT  = pygame.font.SysFont(None, 40)

    # 3) 构造 Operate 所需的“args”占位对象（避免它再次加载 YOLO）
    args = SimpleNamespace(
        ip=ROBOT_IP,
        port=ROBOT_PORT,
        calib_dir="calibration/param/",
        save_data=False,
        play_data=False,
        yolo_model=""   # 空串 => Operate 内不会加载 Detector
    )

    # 4) 创建 Operate 实例（只为获得 _get_robot_pose() 与共享的 PenguinPi）
    operate = op.Operate(args, TITLE_FONT, TEXT_FONT)

    # 5) 直接使用 operate 的相机连接（避免重复连接）
    cam = operate.pibot

    # 6) 我们自己的 YOLO 检测器
    detector = Detector(YOLO_MODEL_PATH)

    print("[yolo_obs_logger] started. LOOP =", LOOP, " PERIOD =", PERIOD_SEC, " trigger_dist =", TRIGGER_DIST)

    def one_pass():
        # a) 获取图像与当前位姿
        img = cam.get_image()
        if img is None:
            return False
        pose = operate._get_robot_pose()  # (x, y, theta)

        # b) YOLO 检测
        try:
            boxes, _ = detector.detect_single_image(img)  # [(label, [cx,cy,w,h]), ...]
        except Exception as e:
            # YOLO 出错不影响系统其他部分
            print("[yolo_obs_logger] detector error:", e)
            return False
        if not boxes:
            return False

        # c) 载入 slam 并尝试写入
        slam_obj = load_slam(SLAM_PATH)
        updated = False

        for label, bbox in boxes:
            lab = (label or "").lower().strip()
            # 购物清单上的类别不作为障碍
            if lab in shopping_set:
                continue
            # 没有尺寸数据，跳过
            dims = DIMS.get(lab)
            if not dims:
                continue

            # 针孔模型估深度，计算与图像中心偏角
            true_h = float(dims[2])
            pix_h  = float(bbox[3])
            cx     = float(bbox[0])

            Z = pinhole_distance(fx, true_h, pix_h)
            theta_cam = img_shift_angle(fx, cx, IMAGE_WIDTH)
            dist_obj = Z / max(1e-6, math.cos(theta_cam))

            # 仅记录 30 cm 内的障碍
            if dist_obj > TRIGGER_DIST:
                continue

            # 机器人坐标系下相对位置 → 世界坐标
            rel_x = dist_obj * math.cos(theta_cam)
            rel_y = dist_obj * math.sin(theta_cam)
            ox, oy = body_to_world(rel_x, rel_y, pose)

            # 合并/写入
            upsert_obstacle(slam_obj, lab, ox, oy)
            updated = True

        if updated:
            save_slam_atomic(SLAM_PATH, slam_obj)
            # 触发重规划
            open("replan.flag", "w").close()
            if GO_HOME_ON_BLOCK:
                open("home.flag", "w").close()
            print("[yolo_obs_logger] slam.txt updated & replan.flag set")
        return updated

    if LOOP:
        while True:
            one_pass()
            time.sleep(PERIOD_SEC)
    else:
        one_pass()


if __name__ == "__main__":
    main()
