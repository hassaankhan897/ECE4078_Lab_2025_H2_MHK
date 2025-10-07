# safe_los_planner.py
import math
import numpy as np
import matplotlib.pyplot as plt

PLANNER_CFG = {
    "arena_half": 1.2,
    "resolution": 0.04,

    # radii (m)
    "robot_radius": 0.08,
    "obstacle_radius": 0.04,
    "inflate_error": 0.04,   # added to obstacle radius only

    # goal behavior
    "goal_tol": 0.18,
    "final_visibility_radius": 0.6,

    # costs (priority order via magnitude)
    "w_aruco_los": 50.0,
    "w_safety":    30.0,
    "w_border":    10.0,
    "w_goal_los":  15.0,

    "border_soft_threshold": 0.15,

    # smoothing
    "extra_clearance": 0.10,
    "waypoint_spacing": 0.10,
}

def _robot_to_plot(xr, yr):  # robot: +x fwd, +y left
    return yr, xr


class SafeLOSPlanner:
    def __init__(self, obstacles, markers, cfg=None):
        self.cfg = dict(PLANNER_CFG)
        if cfg: self.cfg.update(cfg)

        self.obstacles = list(obstacles)   # [(x,y,name)]
        self.markers   = list(markers)     # [(x,y)]

        self.res  = self.cfg["resolution"]
        self.half = self.cfg["arena_half"]
        self.min_x, self.max_x = -self.half, self.half
        self.min_y, self.max_y = -self.half, self.half

        self.xw = int(round((self.max_x - self.min_x)/self.res))
        self.yw = int(round((self.max_y - self.min_y)/self.res))

        # tube for LOS: robot + (obstacle + inflation)
        self.los_tube = self.cfg["robot_radius"] + (self.cfg["obstacle_radius"] + self.cfg["inflate_error"])

        self.motion = [
            ( 1,  0, 1.0), ( 0,  1, 1.0), (-1,  0, 1.0), ( 0, -1, 1.0),
            ( 1,  1, math.sqrt(2)), ( 1, -1, math.sqrt(2)),
            (-1,  1, math.sqrt(2)), (-1, -1, math.sqrt(2)),
        ]

    # ---------- index/pos ----------
    def _pos_from_idx(self, ix, iy):
        return self.min_x + ix*self.res, self.min_y + iy*self.res
    def _idx_from_pos(self, x, y):
        return int(round((x - self.min_x)/self.res)), int(round((y - self.min_y)/self.res))
    def _valid_idx(self, ix, iy):
        return 0 <= ix < self.xw and 0 <= iy < self.yw

    # ---------- keep-out logic ----------
    def _r_keep_for(self, name, target_name, prev_target_name):
        """distance from obstacle center that the ROBOT CENTER must stay outside."""
        rb = self.cfg["robot_radius"]
        bare = self.cfg["obstacle_radius"]
        infl = bare + self.cfg["inflate_error"]

        if target_name and name == target_name:
            return 0.0                    # current target NON-collidable
        if prev_target_name and name == prev_target_name:
            return rb + bare              # previous fruit de-inflated
        return rb + infl                  # everyone else inflated

    def _build_occupancy(self, target_name=None, prev_target_name=None):
        occ = np.zeros((self.xw, self.yw), dtype=bool)
        rb = self.cfg["robot_radius"]

        for ix in range(self.xw):
            x, _ = self._pos_from_idx(ix, 0)
            for iy in range(self.yw):
                _, y = self._pos_from_idx(ix, iy)

                # border keep-out
                if (x < self.min_x + rb or x > self.max_x - rb or
                    y < self.min_y + rb or y > self.max_y - rb):
                    occ[ix, iy] = True
                    continue

                blocked = False
                for (xo, yo, name) in self.obstacles:
                    r_keep = self._r_keep_for(name, target_name, prev_target_name)
                    if r_keep > 0.0 and math.hypot(x - xo, y - yo) <= r_keep:
                        blocked = True
                        break
                occ[ix, iy] = blocked
        return occ

    def _build_clearance(self, target_name=None, prev_target_name=None):
        rb = self.cfg["robot_radius"]
        field = np.full((self.xw, self.yw), np.inf, dtype=float)

        for ix in range(self.xw):
            x, _ = self._pos_from_idx(ix, 0)
            for iy in range(self.yw):
                _, y = self._pos_from_idx(ix, iy)

                d_border = min(
                    abs(x - (self.min_x + rb)),
                    abs(x - (self.max_x - rb)),
                    abs(y - (self.min_y + rb)),
                    abs(y - (self.max_y - rb)),
                )
                ds = [d_border]

                for (xo, yo, name) in self.obstacles:
                    r_keep = self._r_keep_for(name, target_name, prev_target_name)
                    if r_keep == 0.0:
                        continue
                    ds.append(max(0.0, math.hypot(x - xo, y - yo) - r_keep))

                field[ix, iy] = min(ds)
        return field

    # ---------- LOS ----------
    def _segment_clear(self, x1, y1, x2, y2, tube, ignore_names=None):
        ignore = set(ignore_names or [])
        for (ox, oy, name) in self.obstacles:
            if name in ignore:
                continue
            ax, ay = x1, y1; bx, by = x2, y2
            abx, aby = bx-ax, by-ay
            apx, apy = ox-ax, oy-ay
            denom = abx*abx + aby*aby + 1e-12
            t = max(0.0, min(1.0, (apx*abx + apy*aby)/denom))
            cx, cy = ax + t*abx, ay + t*aby
            if math.hypot(ox - cx, oy - cy) <= tube:
                return False
        return True
    def _has_los_to_any_marker(self, x, y):
        for (mx, my) in self.markers:
            if self._segment_clear(x, y, mx, my, self.los_tube):
                return True
        return False
    def _has_los_to_target(self, x, y, gx, gy, target_name=None):
        ignore = [target_name] if target_name else None
        return self._segment_clear(x, y, gx, gy, self.los_tube, ignore_names=ignore)

    # ---------- costs ----------
    def _heuristic(self, ix, iy, gx, gy):
        x, y = self._pos_from_idx(ix, iy)
        return math.hypot(x - gx, y - gy)

    def _cost_terms(self, ix, iy, gx, gy, clearance, target_name=None):
        x, y = self._pos_from_idx(ix, iy)
        cfg = self.cfg

        c_aruco = -cfg["w_aruco_los"] if self._has_los_to_any_marker(x, y) else 0.0

        clr = clearance[ix, iy]
        c_safety = cfg["w_safety"] / (max(clr, 1e-3)**2 + 1e-6)

        border_soft = cfg["border_soft_threshold"]
        dist_to_border = min(
            abs(x - (self.min_x + cfg["robot_radius"])),
            abs(x - (self.max_x - cfg["robot_radius"])),
            abs(y - (self.min_y + cfg["robot_radius"])),
            abs(y - (self.max_y - cfg["robot_radius"])),
        )
        c_border = cfg["w_border"] * (1.0 - dist_to_border / border_soft) if dist_to_border < border_soft else 0.0

        d_goal = math.hypot(x - gx, y - gy)
        if d_goal <= cfg["final_visibility_radius"]:
            c_goalvis = -cfg["w_goal_los"] if self._has_los_to_target(x, y, gx, gy, target_name) else +cfg["w_goal_los"]
        else:
            c_goalvis = 0.0

        return c_aruco + c_safety + c_border + c_goalvis

    class Node:
        __slots__ = ("ix", "iy", "g", "pid")
        def __init__(self, ix, iy, g, pid):
            self.ix, self.iy, self.g, self.pid = ix, iy, g, pid

    # ---------- planning ----------
    def plan(self, sx, sy, gx, gy, target_name=None, prev_target_name=None):
        if math.hypot(sx - gx, sy - gy) <= self.cfg["goal_tol"]:
            return [sx, gx], [sy, gy]

        occ = self._build_occupancy(target_name, prev_target_name)
        clr = self._build_clearance(target_name, prev_target_name)

        s_ix, s_iy = self._idx_from_pos(sx, sy)
        if not self._valid_idx(s_ix, s_iy):
            return [], []

        # allow start if only blocked by previous fruit
        if occ[s_ix, s_iy]:
            allow = False
            if prev_target_name:
                occ_wo_prev = self._build_occupancy(target_name, None)
                if not occ_wo_prev[s_ix, s_iy]:
                    allow = True
            if allow:
                occ[s_ix, s_iy] = False
            else:
                return [], []

        open_set, closed, parent = {}, {}, {}
        sid = (s_ix, s_iy)
        open_set[sid] = self.Node(s_ix, s_iy, 0.0, None)
        parent[sid] = None

        while open_set:
            # best f
            best_id, best_f = None, float("inf")
            for (ix, iy), node in open_set.items():
                f = node.g + self._heuristic(ix, iy, gx, gy) + self._cost_terms(ix, iy, gx, gy, clr, target_name)
                if f < best_f:
                    best_f, best_id = f, (ix, iy)

            curr = open_set.pop(best_id)
            closed[best_id] = curr
            cx, cy = self._pos_from_idx(curr.ix, curr.iy)
            if math.hypot(cx - gx, cy - gy) <= self.cfg["goal_tol"]:
                rx, ry = self._reconstruct_path(parent, best_id)
                # ensure final inside goal_tol
                rx, ry = self._enforce_goal_tolerance(rx, ry, gx, gy, target_name)
                return rx, ry

            for dx, dy, step in self.motion:
                nix, niy = curr.ix + dx, curr.iy + dy
                nid = (nix, niy)
                if not self._valid_idx(nix, niy):        continue
                if occ[nix, niy]:                        continue
                if nid in closed:                        continue

                ng = curr.g + step * self.res
                if nid not in open_set or ng < open_set[nid].g:
                    open_set[nid] = self.Node(nix, niy, ng, best_id)
                    parent[nid] = best_id

        return [], []

    def _reconstruct_path(self, parent, last_id):
        xs, ys = [], []
        nid = last_id
        while nid is not None:
            ix, iy = nid
            x, y = self._pos_from_idx(ix, iy)
            xs.append(x); ys.append(y)
            nid = parent[nid]
        xs.reverse(); ys.reverse()
        return xs, ys

    # ---------- enforce goal tolerance ----------
    def _enforce_goal_tolerance(self, rx, ry, gx, gy, target_name=None):
        if not rx: return rx, ry
        dx, dy = gx - rx[-1], gy - ry[-1]
        d = math.hypot(dx, dy)
        if d <= self.cfg["goal_tol"]:
            return rx, ry

        scale = max(1e-6, d)
        ux, uy = dx/scale, dy/scale
        px, py = gx - ux*(self.cfg["goal_tol"]*0.95), gy - uy*(self.cfg["goal_tol"]*0.95)

        if self._segment_clear(rx[-1], ry[-1], px, py, self.los_tube, ignore_names=[target_name] if target_name else None):
            rx.append(px); ry.append(py)

        return rx, ry

    # ---------- smoothing ----------
    def smooth(self, rx, ry, gx=None, gy=None, target_name=None):
        if not rx: return rx, ry

        def seg_clear(x1, y1, x2, y2):
            return self._segment_clear(x1, y1, x2, y2, self.los_tube + self.cfg["extra_clearance"])

        new_rx, new_ry = [rx[0]], [ry[0]]
        i = 0
        while i < len(rx) - 1:
            j = len(rx) - 1
            while j > i + 1 and not seg_clear(rx[i], ry[i], rx[j], ry[j]):
                j -= 1
            new_rx.append(rx[j]); new_ry.append(ry[j]); i = j

        frx, fry = [new_rx[0]], [new_ry[0]]
        for x, y in zip(new_rx[1:], new_ry[1:]):
            if math.hypot(x - frx[-1], y - fry[-1]) >= self.cfg["waypoint_spacing"]:
                frx.append(x); fry.append(y)

        if gx is not None and gy is not None:
            frx, fry = self._enforce_goal_tolerance(frx, fry, gx, gy, target_name)

        return frx, fry

    # ---------- plotting ----------
    def plot_run(self, rx, ry, obstacles, sx, sy, gx, gy, target_name=None, prev_target_name=None):
        cfg = self.cfg
        infl_r = cfg["obstacle_radius"] + cfg["inflate_error"]

        fig, ax = plt.subplots()
        ax.set_xlim(-cfg["arena_half"], cfg["arena_half"])
        ax.set_ylim(-cfg["arena_half"], cfg["arena_half"])
        ax.set_aspect('equal')
        ax.set_title(f"Safe-LOS A* → {target_name or 'Target'}")
        ax.set_xticks(np.arange(-cfg["arena_half"], cfg["arena_half"] + 1e-9, 0.2))
        ax.set_yticks(np.arange(-cfg["arena_half"], cfg["arena_half"] + 1e-9, 0.2))
        ax.grid(True, which='both', linestyle="--", linewidth=0.5)

        # arena
        ax.add_patch(plt.Rectangle((-cfg["arena_half"], -cfg["arena_half"]),
                                   2*cfg["arena_half"], 2*cfg["arena_half"],
                                   fill=False, linestyle=":", linewidth=0.9, edgecolor="gray",
                                   label="Arena boundary"))

        first_infl = False
        for (x, y, name) in obstacles:
            px, py = _robot_to_plot(x, y)
            is_marker = "aruco" in name.lower()
            color = "blue" if is_marker else "red"

            ax.add_patch(plt.Circle((px, py), cfg["obstacle_radius"], color=color, alpha=0.3))

            if not ((target_name and name == target_name) or (prev_target_name and name == prev_target_name)):
                ax.add_patch(plt.Circle((px, py), infl_r, fill=False, linestyle="--",
                                        linewidth=0.9, edgecolor="orange", alpha=0.85,
                                        label=None if first_infl else "Inflated obstacle"))
                first_infl = True

            if is_marker:
                ax.plot(px, py, "bx", markersize=8, label="ArUco marker")
            ax.text(px, py, name, fontsize=7, color=color, ha="center", va="bottom")

        # start/goal
        sxp, syp = _robot_to_plot(sx, sy)
        ax.plot(sxp, syp, "go", markersize=9, label="Start")

        gxp, gyp = _robot_to_plot(gx, gy)
        ax.plot(gxp, gyp, "ro", markersize=6, label="Target")
        ax.add_patch(plt.Circle((gxp, gyp), cfg["goal_tol"], fill=False, linestyle="-.",
                                linewidth=0.9, edgecolor="purple", alpha=0.9, label="Goal tol"))
        ax.add_patch(plt.Circle((gxp, gyp), cfg["final_visibility_radius"], fill=False, linestyle=":",
                                linewidth=0.8, edgecolor="purple", alpha=0.5, label="Final-vis zone"))

        if rx:
            rxp, ryp = zip(*[_robot_to_plot(x, y) for x, y in zip(rx, ry)])
            ax.plot(rxp, ryp, "-g", linewidth=2.0, label="Planned Path")
            ax.plot(rxp, ryp, "kx", markersize=3)

        ax.arrow(0, 0, 0, 0.3, head_width=0.05, color="red")
        ax.arrow(0, 0, -0.3, 0, head_width=0.05, color="blue")
        ax.text(0.0, 0.34, "+x (forward)", color="red", fontsize=8, ha="center")
        ax.text(-0.34, 0.0, "+y (left)", color="blue", fontsize=8, va="center")
        ax.set_xlabel("+y (left) ←"); ax.set_ylabel("↑ +x (forward)")

        h, l = ax.get_legend_handles_labels()
        ax.legend(dict(zip(l, h)).values(), dict(zip(l, h)).keys(), fontsize=8, loc="lower right")
        plt.show(block=True)
