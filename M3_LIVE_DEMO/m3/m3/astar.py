"""
Path planning test script for PenguinPi grocery shopping
---------------------------------------------------------

This script:
 - Loads SLAM map (slam.txt) and shopping list (shopping_list.txt)
 - Uses a weighted A* algorithm with safety-first cost
 - Inflates obstacles (fruits/vegs and ArUco markers) with clearance radius
 - Plans path to each shopping list item sequentially
 - Smooths path while preserving clearance
 - Visualizes arena, obstacles, targets, paths
 - Prints waypoints for robot navigation
"""

import json
import matplotlib.pyplot as plt
import math
import numpy as np


# -----------------------
# Configurable Parameters
# -----------------------
CONFIG = {
    # Path Planning
    "resolution": 0.02,      # grid size [m]
    "alpha": 50.0,           # safety weighting (higher = safer, longer paths)
    "goal_tol": 0.155,       # tolerance for reaching target [m]

    # Arena
    "arena_half": 1.2,       # half-size of arena [m] (2.4 x 2.4 arena)

    # Robot and Obstacles
    "robot_radius": 0.045,   # robot radius [m]
    "obstacle_radius": 0.045, # fruit/veg & ArUco radius [m]
    "safety_margin": 0.05,   # extra inflation around obstacles [m]

    # Path Smoothing
    "extra_clearance": 0.1,  # extra clearance for shortcut smoothing [m]
    "waypoint_spacing": 0.1, # min spacing between waypoints [m]

    # Visualization
    "grid_interval": 0.2,    # grid spacing in plots [m]
}


# -----------------------
# Coordinate transforms
# -----------------------
def cartesian_to_robot(x_c, y_c):
    """Cartesian (slam.txt) → Robot frame coordinates."""
    x_r = y_c
    y_r = -x_c
    return x_r, y_r


# -----------------------
# Weighted A* Planner (safety first)
# -----------------------
class AStarPlanner:
    def __init__(self, ox, oy,
                 resolution=CONFIG["resolution"],
                 rr=None,
                 alpha=CONFIG["alpha"],
                 goal_tol=CONFIG["goal_tol"],
                 arena_half=CONFIG["arena_half"],
                 robot_radius=CONFIG["robot_radius"]):
        if rr is None:
            rr = CONFIG["robot_radius"] + CONFIG["obstacle_radius"] + CONFIG["safety_margin"]

        self.resolution = resolution
        self.rr = rr
        self.alpha = alpha
        self.goal_tol = goal_tol
        self.arena_half = arena_half
        self.robot_radius = robot_radius

        self.min_x, self.min_y = -arena_half, -arena_half
        self.max_x, self.max_y = arena_half, arena_half

        self.x_width = int(round((self.max_x - self.min_x) / resolution))
        self.y_width = int(round((self.max_y - self.min_y) / resolution))

        self.obstacle_map = self.calc_obstacle_map(ox, oy)
        self.dist_map = self.calc_dist_map(ox, oy)
        self.motion = self.get_motion_model()

    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x
            self.y = y
            self.cost = cost
            self.parent_index = parent_index
        def __lt__(self, other): return self.cost < other.cost

    def planning(self, sx, sy, gx, gy):
        """Plan path from start (sx,sy) to goal (gx,gy)."""
        start_node = self.Node(self.calc_xy_index(sx, self.min_x),
                               self.calc_xy_index(sy, self.min_y), 0.0, -1)

        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(start_node)] = start_node

        while open_set:
            # Pick node with min (cost + heuristic)
            c_id = min(open_set, key=lambda o: open_set[o].cost +
                       self.calc_heuristic(open_set[o], gx, gy))
            current = open_set[c_id]

            cx = self.calc_position(current.x, self.min_x)
            cy = self.calc_position(current.y, self.min_y)

            # Goal reached within tolerance
            if math.hypot(cx - gx, cy - gy) <= self.goal_tol:
                return self.calc_final_path(current, closed_set)

            del open_set[c_id]
            closed_set[c_id] = current

            for move_x, move_y, move_cost in self.motion:
                node = self.Node(current.x + move_x,
                                 current.y + move_y,
                                 current.cost + move_cost, c_id)

                n_id = self.calc_grid_index(node)
                if not self.verify_node(node): 
                    continue
                if n_id in closed_set: 
                    continue

                px = self.calc_position(node.x, self.min_x)
                py = self.calc_position(node.y, self.min_y)
                dist = self.dist_map[node.x][node.y]

                # SAFETY-FIRST COSTING (inverse-square penalty)
                penalty = self.alpha / (dist**2 + 1e-3)
                node.cost += penalty

                if n_id not in open_set or open_set[n_id].cost > node.cost:
                    open_set[n_id] = node

        return [], []

    def calc_final_path(self, goal_node, closed_set):
        """Trace back from goal to start."""
        rx, ry = [self.calc_position(goal_node.x, self.min_x)], [
            self.calc_position(goal_node.y, self.min_y)]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            node = closed_set[parent_index]
            rx.append(self.calc_position(node.x, self.min_x))
            ry.append(self.calc_position(node.y, self.min_y))
            parent_index = node.parent_index
        return rx[::-1], ry[::-1]

    def calc_position(self, index, minp): 
        return index * self.resolution + minp

    def calc_xy_index(self, position, minp): 
        return int(round((position - minp) / self.resolution))

    def calc_grid_index(self, node): 
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def calc_heuristic(self, node, gx, gy):
        """Euclidean distance heuristic."""
        px = self.calc_position(node.x, self.min_x)
        py = self.calc_position(node.y, self.min_y)
        return math.hypot(px - gx, py - gy)

    def verify_node(self, node):
        """Check arena boundaries and obstacle collisions."""
        px, py = self.calc_position(node.x, self.min_x), self.calc_position(node.y, self.min_y)

        if px < self.min_x + self.robot_radius or py < self.min_y + self.robot_radius:
            return False
        if px > self.max_x - self.robot_radius or py > self.max_y - self.robot_radius:
            return False

        if self.obstacle_map[node.x][node.y]: 
            return False
        return True

    def calc_obstacle_map(self, ox, oy):
        """Build inflated obstacle map."""
        obstacle_map = [[False for _ in range(self.y_width)] for _ in range(self.x_width)]
        for ix in range(self.x_width):
            x = self.calc_position(ix, self.min_x)
            for iy in range(self.y_width):
                y = self.calc_position(iy, self.min_y)
                for (ox_, oy_) in zip(ox, oy):
                    if math.hypot(ox_ - x, oy_ - y) <= self.rr:
                        obstacle_map[ix][iy] = True
                        break
        return obstacle_map

    def calc_dist_map(self, ox, oy):
        """Compute clearance distance map."""
        dist_map = [[float("inf") for _ in range(self.y_width)] for _ in range(self.x_width)]
        for ix in range(self.x_width):
            x = self.calc_position(ix, self.min_x)
            for iy in range(self.y_width):
                y = self.calc_position(iy, self.min_y)

                dists = [math.hypot(ox_ - x, oy_ - y) for (ox_, oy_) in zip(ox, oy)]
                dists.append(abs(x - (self.min_x + self.robot_radius)))
                dists.append(abs(x - (self.max_x - self.robot_radius)))
                dists.append(abs(y - (self.min_y + self.robot_radius)))
                dists.append(abs(y - (self.max_y - self.robot_radius)))

                dist_map[ix][iy] = min(dists) if dists else float("inf")
        return dist_map

    @staticmethod
    def get_motion_model():
        """8-connected grid movement model."""
        return [[1, 0, 1], [0, 1, 1], [-1, 0, 1], [0, -1, 1],
                [-1, -1, math.sqrt(2)], [-1, 1, math.sqrt(2)],
                [1, -1, math.sqrt(2)], [1, 1, math.sqrt(2)]]


# -----------------------
# Path Smoothing (safe)
# -----------------------
def smooth_path(rx, ry, ox, oy, rr,
                extra_clearance=CONFIG["extra_clearance"],
                waypoint_spacing=CONFIG["waypoint_spacing"]):
    """Shortcut-based path smoothing with clearance checks."""
    if not rx or not ry:
        return rx, ry

    def is_collision(x1, y1, x2, y2):
        for (ox_, oy_) in zip(ox, oy):
            A = np.array([x1, y1]); B = np.array([x2, y2]); P = np.array([ox_, oy_])
            AB = B - A
            t = np.clip(np.dot(P - A, AB) / (np.dot(AB, AB) + 1e-6), 0.0, 1.0)
            closest = A + t * AB
            dist = np.linalg.norm(P - closest)
            if dist <= rr + extra_clearance:
                return True
        return False

    new_rx, new_ry = [rx[0]], [ry[0]]
    i = 0
    while i < len(rx) - 1:
        j = len(rx) - 1
        while j > i + 1:
            if not is_collision(rx[i], ry[i], rx[j], ry[j]):
                break
            j -= 1
        new_rx.append(rx[j]); new_ry.append(ry[j])
        i = j

    filtered_rx, filtered_ry = [new_rx[0]], [new_ry[0]]
    for x, y in zip(new_rx[1:], new_ry[1:]):
        if math.hypot(x - filtered_rx[-1], y - filtered_ry[-1]) >= waypoint_spacing:
            filtered_rx.append(x); filtered_ry.append(y)
    return filtered_rx, filtered_ry


# -----------------------
# Test script
# -----------------------
if __name__ == "__main__":
    with open("slam.txt", "r") as f:
        slam_map = json.load(f)
    with open("shopping_list.txt", "r") as f:
        shopping_list = [line.strip() for line in f.readlines()]

    # Separate landmarks
    fruits_vegs, arucos = [], []
    for name, pos in slam_map.items():
        x_r, y_r = cartesian_to_robot(pos["x"], pos["y"])
        if "aruco" in name:
            arucos.append((x_r, y_r, name))
        else:
            fruits_vegs.append((x_r, y_r, name))

    # Parameters
    inflation = CONFIG["robot_radius"] + CONFIG["obstacle_radius"] + CONFIG["safety_margin"]
    resolution = CONFIG["resolution"]

    sx, sy = 0.0, 0.0  # robot starts at origin

    for target in shopping_list:
        # Partition obstacles vs current goal
        obstacles = []
        gx, gy, target_name = None, None, None
        for (x, y, name) in fruits_vegs + arucos:
            if target in name:
                gx, gy, target_name = x, y, name
            else:
                obstacles.append((x, y, name))

        ox = [o[0] for o in obstacles]
        oy = [o[1] for o in obstacles]

        print(f"\nPlanning path to {target_name}...")

        planner = AStarPlanner(ox, oy, resolution=resolution, rr=inflation)
        rx, ry = planner.planning(sx, sy, gx, gy)
        rx, ry = smooth_path(rx, ry, ox, oy, rr=inflation)

        if rx and ry:
            print(f"Waypoints to {target_name}:")
            for wx, wy in zip(rx, ry):
                print(f"  ({wx:.2f}, {wy:.2f})")
        else:
            print(f"⚠️ No path found to {target_name}")

        # ---- Plot ----
        fig, ax = plt.subplots()
        ax.set_xlim(-CONFIG["arena_half"], CONFIG["arena_half"])
        ax.set_ylim(-CONFIG["arena_half"], CONFIG["arena_half"])
        ax.set_aspect('equal')
        ax.set_title(f"Arena Path Planning to {target_name}")
        ax.set_xticks(np.arange(-CONFIG["arena_half"], CONFIG["arena_half"]+0.01, CONFIG["grid_interval"]))
        ax.set_yticks(np.arange(-CONFIG["arena_half"], CONFIG["arena_half"]+0.01, CONFIG["grid_interval"]))
        ax.grid(True, which='both', linestyle="--", linewidth=0.5)

        # Obstacles
        for (x, y, name) in obstacles:
            color = "red" if "aruco" not in name else "blue"
            circle = plt.Circle((x, y), CONFIG["obstacle_radius"], color=color, alpha=0.3)
            ax.add_patch(circle)
            if "aruco" in name:
                ax.plot(x, y, "bx", markersize=8, label="ArUco")
            ax.text(x, y, name, fontsize=6, color=color, ha="center")

        # Target
        if gx is not None:
            ax.plot(gx, gy, "ro", markersize=6, label="Target")
            ax.text(gx, gy, target_name.split("_")[0], fontsize=8, color="black")

        # Start
        ax.plot(sx, sy, "go", markersize=10, label="Start")

        # Path
        if rx and ry:
            ax.plot(rx, ry, "-g", linewidth=2, label="Planned Path")
            ax.plot(rx, ry, "kx", markersize=4)

        # Deduplicate legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=7)

        plt.show()

        # Update start for next leg
        if rx and ry:
            sx, sy = rx[-1], ry[-1]
