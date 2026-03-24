"""
drone_env.py — Realistic 3D Drone Navigation Environment
=========================================================
World   : 50 m × 50 m × 20 m continuous space
Action  : continuous [vx, vy, vz] ∈ [-1, 1]  →  scaled to MAX_SPEED m/step
Obs (20): goal_dir(3) | goal_dist(1) | pos(3) | vel(3) | lidar×10
Obstacles: buildings (AABB), trees (cylinder), rocks (sphere)  — mixed every episode
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class DroneEnv(gym.Env):
    metadata = {"render_modes": []}

    # ── World constants ──────────────────────────────────────────────
    WORLD      = np.array([50.0, 50.0, 20.0], dtype=np.float64)
    MAX_SPEED  = 1.5          # metres per step
    LIDAR_MAX  = 12.0         # metres
    DRONE_R    = 0.40         # collision radius (metres)
    GOAL_R     = 1.50         # goal-reached radius (metres)
    MIN_HEIGHT = 0.30         # ground clearance

    # 10 lidar directions: 6 cardinal + 4 horizontal diagonals
    _RAW_DIRS = np.array([
        [ 1,  0,  0], [-1,  0,  0],
        [ 0,  1,  0], [ 0, -1,  0],
        [ 0,  0,  1], [ 0,  0, -1],
        [ 1,  1,  0], [-1,  1,  0],
        [ 1, -1,  0], [-1, -1,  0],
    ], dtype=np.float64)

    def __init__(self):
        super().__init__()

        # Normalise diagonal rays to unit length
        norms = np.linalg.norm(self._RAW_DIRS, axis=1, keepdims=True)
        self.LIDAR_DIRS = (self._RAW_DIRS / norms).astype(np.float64)

        # All observations normalised into roughly [-1, 1]
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(20,), dtype=np.float32
        )

        # Continuous 3-axis velocity command
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        self.max_steps  = 600
        self.obstacles  = []
        self.drone_pos  = np.zeros(3, dtype=np.float32)
        self.drone_vel  = np.zeros(3, dtype=np.float32)
        self.goal       = np.zeros(3, dtype=np.float32)
        self._prev_dist = 0.0
        self.steps      = 0

    # ════════════════════════════════════════════════════════════════
    #  RESET
    # ════════════════════════════════════════════════════════════════
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)

        # Drone starts in the low-x / low-y corner of the world
        self.drone_pos = np.array([
            rng.uniform(2.0, 12.0),
            rng.uniform(2.0, 12.0),
            rng.uniform(2.0,  8.0),
        ], dtype=np.float32)

        self.drone_vel = np.zeros(3, dtype=np.float32)

        # Goal is placed in the opposite (high-x / high-y) corner
        self.goal = np.array([
            rng.uniform(38.0, 48.0),
            rng.uniform(38.0, 48.0),
            rng.uniform( 2.0, 12.0),
        ], dtype=np.float32)

        self._generate_obstacles(rng)
        self._clear_near(self.drone_pos, clear_r=4.0)
        self._clear_near(self.goal,      clear_r=4.0)

        self.steps      = 0
        self._prev_dist = float(np.linalg.norm(self.goal - self.drone_pos))

        return self._obs(), {}

    # ════════════════════════════════════════════════════════════════
    #  OBSTACLE GENERATION — mixed shapes every episode
    # ════════════════════════════════════════════════════════════════
    def _generate_obstacles(self, rng: np.random.Generator):
        self.obstacles = []

        # ── Buildings (axis-aligned boxes) ──────────────────────────
        for _ in range(8):
            w = float(rng.uniform(2.5, 6.0))
            h = float(rng.uniform(5.0, 16.0))
            x = float(rng.uniform(10.0, 40.0))
            y = float(rng.uniform(10.0, 40.0))
            self.obstacles.append({
                "type": "box",
                "pos":  np.array([x, y, h / 2], dtype=np.float32),
                "he":   np.array([w / 2, w / 2, h / 2], dtype=np.float32),
                # ── Visualiser hints ────────────────────────────────
                "vis_w": w, "vis_h": h,
            })

        # ── Trees (vertical cylinders) ──────────────────────────────
        for _ in range(12):
            r = float(rng.uniform(0.30, 0.90))
            h = float(rng.uniform(4.0, 10.0))
            x = float(rng.uniform(3.0, 47.0))
            y = float(rng.uniform(3.0, 47.0))
            self.obstacles.append({
                "type":   "cylinder",
                "pos":    np.array([x, y, h / 2], dtype=np.float32),
                "radius": r,
                "height": h,
            })

        # ── Rocks / boulders (spheres) ──────────────────────────────
        for _ in range(6):
            r = float(rng.uniform(0.60, 2.50))
            x = float(rng.uniform(3.0, 47.0))
            y = float(rng.uniform(3.0, 47.0))
            self.obstacles.append({
                "type":   "sphere",
                "pos":    np.array([x, y, r], dtype=np.float32),   # sits on ground
                "radius": r,
            })

    def _clear_near(self, pt: np.ndarray, clear_r: float):
        """Remove any obstacle whose centre is within *clear_r* of *pt*."""
        self.obstacles = [
            o for o in self.obstacles
            if float(np.linalg.norm(o["pos"] - pt)) > clear_r
        ]

    # ════════════════════════════════════════════════════════════════
    #  COLLISION DETECTION  (per-shape exact tests)
    # ════════════════════════════════════════════════════════════════
    def _collides(self) -> bool:
        p = self.drone_pos
        r = self.DRONE_R

        if p[2] < self.MIN_HEIGHT:
            return True                          # ground hit

        for o in self.obstacles:
            if o["type"] == "sphere":
                if np.linalg.norm(p - o["pos"]) < o["radius"] + r:
                    return True

            elif o["type"] == "box":
                closest = np.clip(p, o["pos"] - o["he"], o["pos"] + o["he"])
                if np.linalg.norm(p - closest) < r:
                    return True

            elif o["type"] == "cylinder":
                xy_d = np.linalg.norm(p[:2] - o["pos"][:2])
                z_lo = o["pos"][2] - o["height"] / 2 - r
                z_hi = o["pos"][2] + o["height"] / 2 + r
                if xy_d < o["radius"] + r and z_lo < p[2] < z_hi:
                    return True
        return False

    # ════════════════════════════════════════════════════════════════
    #  RAY – SHAPE INTERSECTIONS  (accurate geometry)
    # ════════════════════════════════════════════════════════════════
    @staticmethod
    def _ray_sphere(ro, rd, c, r):
        """Distance along ray to sphere surface, or None if no hit."""
        oc   = ro - c
        b    = 2.0 * np.dot(oc, rd)
        disc = b * b - 4.0 * (np.dot(oc, oc) - r * r)
        if disc < 0:
            return None
        t = (-b - np.sqrt(disc)) / 2.0
        return float(t) if t > 1e-4 else None

    @staticmethod
    def _ray_box(ro, rd, c, he):
        """Distance along ray to AABB surface, or None if no hit."""
        lo, hi = c - he, c + he
        tmin, tmax = -1e9, 1e9
        for i in range(3):
            if abs(rd[i]) < 1e-8:
                if ro[i] < lo[i] or ro[i] > hi[i]:
                    return None
            else:
                t1 = (lo[i] - ro[i]) / rd[i]
                t2 = (hi[i] - ro[i]) / rd[i]
                if t1 > t2:
                    t1, t2 = t2, t1
                tmin = max(tmin, t1)
                tmax = min(tmax, t2)
                if tmin > tmax:
                    return None
        if tmax < 1e-4:
            return None
        return float(tmin) if tmin > 1e-4 else float(tmax)

    @staticmethod
    def _ray_cylinder(ro, rd, c, r, h):
        """Distance along ray to vertical cylinder, or None if no hit."""
        ox, oy = ro[0] - c[0], ro[1] - c[1]
        dx, dy = rd[0], rd[1]
        z_lo, z_hi = c[2] - h / 2, c[2] + h / 2
        a = dx * dx + dy * dy

        if a < 1e-8:                             # ray is perfectly vertical
            if ox * ox + oy * oy >= r * r:
                return None
            if abs(rd[2]) < 1e-8:
                return None
            t1 = (z_lo - ro[2]) / rd[2]
            t2 = (z_hi - ro[2]) / rd[2]
            t = min(t1, t2) if min(t1, t2) > 1e-4 else max(t1, t2)
            return float(t) if t > 1e-4 else None

        b    = 2.0 * (ox * dx + oy * dy)
        cc   = ox * ox + oy * oy - r * r
        disc = b * b - 4.0 * a * cc
        if disc < 0:
            return None
        sq = np.sqrt(disc)
        for t in [(-b - sq) / (2 * a), (-b + sq) / (2 * a)]:
            if t > 1e-4:
                hz = ro[2] + t * rd[2]
                if z_lo <= hz <= z_hi:
                    return float(t)
        return None

    # ════════════════════════════════════════════════════════════════
    #  LIDAR  (10-direction ray casting)
    # ════════════════════════════════════════════════════════════════
    def _lidar(self) -> np.ndarray:
        ro     = self.drone_pos.astype(np.float64)
        r_pad  = self.DRONE_R          # inflate obstacles for safety margin
        result = np.ones(10, dtype=np.float32)   # 1.0 = max range (normalised)

        for idx, rd in enumerate(self.LIDAR_DIRS):
            best = self.LIDAR_MAX
            for o in self.obstacles:
                if o["type"] == "sphere":
                    t = self._ray_sphere(ro, rd, o["pos"], o["radius"] + r_pad)
                elif o["type"] == "box":
                    t = self._ray_box(ro, rd, o["pos"], o["he"] + r_pad)
                elif o["type"] == "cylinder":
                    t = self._ray_cylinder(ro, rd, o["pos"],
                                           o["radius"] + r_pad, o["height"])
                else:
                    t = None
                if t is not None and 0 < t < best:
                    best = t
            result[idx] = float(best) / self.LIDAR_MAX   # normalise → [0, 1]

        return result

    # ════════════════════════════════════════════════════════════════
    #  OBSERVATION  (all values ∈ [-1, 1] or [0, 1])
    # ════════════════════════════════════════════════════════════════
    def _obs(self) -> np.ndarray:
        dv       = (self.goal - self.drone_pos).astype(np.float64)
        dist     = float(np.linalg.norm(dv))
        goal_dir = (dv / (dist + 1e-8)).astype(np.float32)          # unit vec
        goal_dist_n = np.float32(np.clip(dist / 70.0, 0.0, 1.0))   # ~max diagonal

        pos_n = (self.drone_pos / self.WORLD).astype(np.float32)                   # [0,1]
        vel_n = np.clip(self.drone_vel / self.MAX_SPEED, -1.0, 1.0).astype(np.float32)

        lidar = self._lidar()

        return np.concatenate([goal_dir, [goal_dist_n], pos_n, vel_n, lidar])

    # ════════════════════════════════════════════════════════════════
    #  STEP
    # ════════════════════════════════════════════════════════════════
    def step(self, action):
        action          = np.clip(action, -1.0, 1.0).astype(np.float32)
        self.drone_vel  = action * self.MAX_SPEED
        self.drone_pos  = self.drone_pos + self.drone_vel

        # ── Soft boundary: clip and zero the offending velocity axis ─
        boundary_hit = False
        lo = np.array([0.0, 0.0, self.MIN_HEIGHT], dtype=np.float32)
        hi = self.WORLD.astype(np.float32)
        clipped = np.clip(self.drone_pos, lo, hi)
        if not np.array_equal(clipped, self.drone_pos):
            boundary_hit     = True
            self.drone_pos   = clipped
            self.drone_vel   = np.zeros(3, dtype=np.float32)

        self.steps += 1

        # ── Collision check ──────────────────────────────────────────
        if self._collides():
            return self._obs(), -100.0, True, False, {"reason": "collision"}

        # ── Reward ───────────────────────────────────────────────────
        new_dist = float(np.linalg.norm(self.goal - self.drone_pos))
        reward   = (self._prev_dist - new_dist) * 3.0   # scaled distance progress
        reward  -= 0.05                                   # per-step time penalty
        if boundary_hit:
            reward -= 5.0

        # Proximity-to-obstacle penalty (danger zone < 2.5 m)
        if self.obstacles:
            min_obs = float(min(
                np.linalg.norm(self.drone_pos - o["pos"]) for o in self.obstacles
            ))
            if min_obs < 2.5:
                reward -= (2.5 - min_obs) * 0.5

        self._prev_dist = new_dist

        # ── Goal reached ─────────────────────────────────────────────
        if new_dist < self.GOAL_R:
            return self._obs(), 200.0, True, False, {"reason": "goal"}

        done = self.steps >= self.max_steps
        return self._obs(), reward, done, False,\
               {"reason": "timeout" if done else "running"}
