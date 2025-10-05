# occupancy_map.py
"""
Occupancy Map (Standalone Viewer + Loader)
------------------------------------------
- Reads SLAM map (slam.txt) with positions in ROBOT frame: +x forward, +y left.
- Visualizes arena, obstacles (fruits+markers), and *inflated obstacle radius only*.
- Provides load_slam_map() so the planner can reuse the same map.

Run:
    python occupancy_map.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt

MAP_CFG = {
    "arena_half": 1.2,          # 2.4 m square arena
    "obstacle_radius": 0.04,    # fruit/aruco geometry radius [m]
    "robot_radius": 0.18,       # robot physical radius [m] (used by planner, not drawn per obstacle)
    "inflate_error": 0.05,      # extra inflation added to obstacle radius [m]
    "resolution": 0.05,         # grid (for display ticks only)
}

def _robot_to_plot(x_robot, y_robot):
    """Robot coords (+x forward, +y left) → plot coords."""
    return y_robot, x_robot

def load_slam_map(filename="slam.txt"):
    """
    Returns:
        ox, oy: lists of obstacle x/y
        obstacles: [(x, y, name), ...]
        markers:   [(x, y), ...] subset where "aruco" in name
    """
    with open(filename, "r") as f:
        slam_map = json.load(f)

    ox, oy, obstacles, markers = [], [], [], []
    for name, pos in slam_map.items():
        x, y = float(pos["x"]), float(pos["y"])
        ox.append(x)
        oy.append(y)
        obstacles.append((x, y, name))
        if "aruco" in name.lower():
            markers.append((x, y))
    return ox, oy, obstacles, markers

def plot_occupancy_map(ox, oy, obstacles, markers, cfg=MAP_CFG):
    # for visualization: show *inflated obstacle only* (obstacle_radius + inflate_error)
    inflated_obstacle_r = cfg["obstacle_radius"] + cfg["inflate_error"]

    fig, ax = plt.subplots()
    ax.set_xlim(-cfg["arena_half"], cfg["arena_half"])
    ax.set_ylim(-cfg["arena_half"], cfg["arena_half"])
    ax.set_aspect('equal')
    ax.set_title("Occupancy Map (Robot Frame)")
    ax.set_xticks(np.arange(-cfg["arena_half"], cfg["arena_half"] + 1e-9, 0.2))
    ax.set_yticks(np.arange(-cfg["arena_half"], cfg["arena_half"] + 1e-9, 0.2))
    ax.grid(True, which='both', linestyle="--", linewidth=0.5)

    # Arena boundary
    arena = plt.Rectangle(
        (-cfg["arena_half"], -cfg["arena_half"]),
        2*cfg["arena_half"], 2*cfg["arena_half"],
        fill=False, linestyle=":", linewidth=0.9, edgecolor="gray",
        label="Arena boundary"
    )
    ax.add_patch(arena)

    # Obstacles (fruits + aruco)
    first_inflated = False
    for (x, y, name) in obstacles:
        px, py = _robot_to_plot(x, y)
        is_marker = "aruco" in name.lower()
        color = "blue" if is_marker else "red"

        # Physical object circle (visual only)
        ax.add_patch(plt.Circle((px, py), cfg["obstacle_radius"], color=color, alpha=0.3))

        # Inflated obstacle (obstacle + error) — this is NOT robot+obstacle
        ax.add_patch(plt.Circle(
            (px, py), inflated_obstacle_r, fill=False, linestyle="--", linewidth=0.9,
            edgecolor="orange", alpha=0.85,
            label=None if first_inflated else "Inflated obstacle"
        ))
        first_inflated = True

        if is_marker:
            ax.plot(px, py, "bx", markersize=8, label="ArUco marker")

        ax.text(px, py, name, fontsize=7, color=color, ha="center", va="bottom")

    # Robot axes annotation
    ax.arrow(0, 0, 0, 0.3, head_width=0.05, color="red")
    ax.arrow(0, 0, -0.3, 0, head_width=0.05, color="blue")
    ax.text(0.0, 0.34, "+x (forward)", color="red", fontsize=8, ha="center")
    ax.text(-0.34, 0.0, "+y (left)", color="blue", fontsize=8, va="center")

    ax.set_xlabel("+y (left) ←")
    ax.set_ylabel("↑ +x (forward)")

    # Legend dedup
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=8, loc="lower right")

    plt.show(block=True)

if __name__ == "__main__":
    ox, oy, obstacles, markers = load_slam_map("slam.txt")
    plot_occupancy_map(ox, oy, obstacles, markers, MAP_CFG)
