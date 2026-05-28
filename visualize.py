"""
visualize.py — PyBullet Visualization of the Trained Drone Agent
=================================================================
Features
  • Quadrotor drone  : body + 2 crossed arms + 4 spinning rotors + landing legs
  • Realistic scene  : buildings (varied colours), trees (trunk + canopy),
                       rocks (grey/brown spheres), grass-coloured ground
  • Flight trail     : cyan line that fades with age
  • Follow camera    : smooth camera tracks the drone
  • HUD              : episode counter, distance to goal, status overlay
  • Clean exit       : Ctrl-C or closing the window terminates gracefully
"""
print("Script started")
import sys
import time
import warnings
import numpy as np
import pybullet as p
import pybullet_data

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from drone_env import DroneEnv

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def quat_z(angle_deg: float):
    """Quaternion for a rotation *angle_deg* around the Z axis → [x,y,z,w]."""
    a = np.radians(angle_deg) / 2.0
    return [0.0, 0.0, float(np.sin(a)), float(np.cos(a))]


def quat_x(angle_deg: float):
    """Quaternion for a rotation *angle_deg* around the X axis → [x,y,z,w]."""
    a = np.radians(angle_deg) / 2.0
    return [float(np.sin(a)), 0.0, 0.0, float(np.cos(a))]


# ═══════════════════════════════════════════════════════════════════════════
#  PyBullet SETUP
# ═══════════════════════════════════════════════════════════════════════════
client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, 0)
p.setRealTimeSimulation(0)

# Hide side panels + disable unnecessary rendering features for performance
p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS,           1)
p.configureDebugVisualizer(p.COV_ENABLE_GUI,               0)
p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW,0)
p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING,     1)
p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)

p.resetDebugVisualizerCamera(
    cameraDistance     = 35,
    cameraYaw          = 45,
    cameraPitch        = -30,
    cameraTargetPosition = [25, 25, 5],
)

# Ground — grass green
plane = p.loadURDF("plane.urdf")
p.changeVisualShape(plane, -1, rgbaColor=[0.35, 0.52, 0.25, 1.0])


# ═══════════════════════════════════════════════════════════════════════════
#  LOAD MODEL  +  ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════════════
viz_env = DummyVecEnv([DroneEnv])
viz_env = VecNormalize.load("vec_normalize.pkl", viz_env)
viz_env.training    = False
viz_env.norm_reward = False

model   = PPO.load("drone_model.zip", env=viz_env)
raw_env = viz_env.envs[0]          # direct access to DroneEnv instance
obs     = viz_env.reset()


# ═══════════════════════════════════════════════════════════════════════════
#  DRONE  ─  compound quadrotor shape
# ═══════════════════════════════════════════════════════════════════════════
S       = 1.0           # drone scale (metres)
DARK    = [0.12, 0.12, 0.12, 1.0]
MID     = [0.25, 0.25, 0.25, 1.0]
RED_ACC = [0.90, 0.15, 0.15, 1.0]
YELLOW  = [0.98, 0.78, 0.05, 1.0]

drone_parts: dict[str, int] = {}


def _make_body(vis_idx, pos, orn=None):
    orn = orn or [0, 0, 0, 1]
    return p.createMultiBody(0, -1, vis_idx, pos, orn)


def build_drone():
    """Construct all drone parts and store their body IDs."""
    global drone_parts
    for bid in drone_parts.values():
        p.removeBody(bid)
    drone_parts.clear()

    o = [0.0, 0.0, 1.0]   # default origin; will be moved every step

    # ── Central body (flat box) ──────────────────────────────────────
    vis = p.createVisualShape(p.GEOM_BOX,
                              halfExtents=[0.20*S, 0.20*S, 0.055*S],
                              rgbaColor=DARK)
    drone_parts["body"] = _make_body(vis, o)

    # ── Two crossed arms (45° and −45°) ─────────────────────────────
    arm_he = [0.30*S, 0.040*S, 0.030*S]
    for i, ang in enumerate([45.0, -45.0]):
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=arm_he, rgbaColor=MID)
        drone_parts[f"arm_{i}"] = _make_body(vis, o, quat_z(ang))

    # ── 4 Rotors — flat cylinders at arm tips ───────────────────────
    ROTOR_OFF = 0.305 * S            # distance from body centre to motor
    _rotor_local = [
        [ ROTOR_OFF,  ROTOR_OFF, 0.06*S],
        [-ROTOR_OFF,  ROTOR_OFF, 0.06*S],
        [-ROTOR_OFF, -ROTOR_OFF, 0.06*S],
        [ ROTOR_OFF, -ROTOR_OFF, 0.06*S],
    ]
    for i, lp in enumerate(_rotor_local):
        vis = p.createVisualShape(p.GEOM_CYLINDER,
                                  radius=0.140*S, length=0.018*S,
                                  rgbaColor=[0.05, 0.05, 0.05, 0.85])
        drone_parts[f"rotor_{i}"] = _make_body(vis, [lp[0], lp[1], 1.0])

    # ── Motor hub nubs (tiny spheres above each rotor) ───────────────
    for i, lp in enumerate(_rotor_local):
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.040*S, rgbaColor=MID)
        drone_parts[f"hub_{i}"] = _make_body(vis, [lp[0], lp[1], 1.0 + 0.04*S])

    # ── 4 Landing legs ───────────────────────────────────────────────
    leg_he   = [0.025*S, 0.025*S, 0.12*S]
    leg_offsets = [
        [ 0.14*S,  0.14*S, -0.10*S],
        [-0.14*S,  0.14*S, -0.10*S],
        [-0.14*S, -0.14*S, -0.10*S],
        [ 0.14*S, -0.14*S, -0.10*S],
    ]
    for i, lp in enumerate(leg_offsets):
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=leg_he, rgbaColor=MID)
        drone_parts[f"leg_{i}"] = _make_body(vis, [lp[0], lp[1], 1.0 + lp[2]])

    # ── Front direction indicator (red dot) ─────────────────────────
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.050*S, rgbaColor=RED_ACC)
    drone_parts["fwd"] = _make_body(vis, [0.21*S, 0.0, 1.0])


# Offsets used in update_drone (relative to body centre)
_ARM_ANGS    = [45.0, -45.0]
_ROTOR_REL   = np.array([
    [ 0.305,  0.305,  0.06],
    [-0.305,  0.305,  0.06],
    [-0.305, -0.305,  0.06],
    [ 0.305, -0.305,  0.06],
]) * S
_HUB_REL     = _ROTOR_REL + np.array([0, 0, 0.04*S])
_LEG_REL     = np.array([
    [ 0.14,  0.14, -0.10],
    [-0.14,  0.14, -0.10],
    [-0.14, -0.14, -0.10],
    [ 0.14, -0.14, -0.10],
]) * S
_ROTOR_DIRS  = [1, -1, 1, -1]   # alternating CW/CCW spin

def update_drone(pos: np.ndarray, orn: list, rotor_ang: float):
    pos_list = [float(pos[0]), float(pos[1]), float(pos[2])]

    # Body
    p.resetBasePositionAndOrientation(drone_parts["body"], pos_list, orn)

    # Arms
    for i, ang in enumerate(_ARM_ANGS):
        apos, aorn = p.multiplyTransforms(pos_list, orn, [0, 0, 0], quat_z(ang))
        p.resetBasePositionAndOrientation(drone_parts[f"arm_{i}"], apos, aorn)

    # Rotors + hubs + legs
    for i in range(4):
        # Rotor
        spin_orn = quat_z((rotor_ang * _ROTOR_DIRS[i]) % 360)
        rpos, rorn = p.multiplyTransforms(pos_list, orn, _ROTOR_REL[i].tolist(), spin_orn)
        p.resetBasePositionAndOrientation(drone_parts[f"rotor_{i}"], rpos, rorn)

        # Hub
        hpos, horn = p.multiplyTransforms(pos_list, orn, _HUB_REL[i].tolist(), [0, 0, 0, 1])
        p.resetBasePositionAndOrientation(drone_parts[f"hub_{i}"], hpos, horn)

        # Leg
        lpos, lorn = p.multiplyTransforms(pos_list, orn, _LEG_REL[i].tolist(), [0, 0, 0, 1])
        p.resetBasePositionAndOrientation(drone_parts[f"leg_{i}"], lpos, lorn)

    # Front indicator
    fpos, forn = p.multiplyTransforms(pos_list, orn, [0.21 * S, 0.0, 0.0], [0, 0, 0, 1])
    p.resetBasePositionAndOrientation(drone_parts["fwd"], fpos, forn)

# ═══════════════════════════════════════════════════════════════════════════
#  GOAL  ─  glowing landing-pad ring
# ═══════════════════════════════════════════════════════════════════════════
_goal_sphere = p.createVisualShape(p.GEOM_SPHERE,
                                   radius=1.5,
                                   rgbaColor=[0.0, 1.0, 0.0, 0.25])
_goal_ring   = p.createVisualShape(p.GEOM_CYLINDER,
                                   radius=1.8, length=0.12,
                                   rgbaColor=[0.0, 1.0, 0.2, 0.70])
_goal_post   = p.createVisualShape(p.GEOM_CYLINDER,
                                   radius=0.08, length=2.0,
                                   rgbaColor=[0.9, 0.8, 0.0, 1.0])

goal_sphere_id = p.createMultiBody(0, -1, _goal_sphere, [0, 0, 1])
goal_ring_id   = p.createMultiBody(0, -1, _goal_ring,   [0, 0, 1])
goal_post_id   = p.createMultiBody(0, -1, _goal_post,   [0, 0, 1])


def update_goal(gpos: np.ndarray):
    gx, gy, gz = float(gpos[0]), float(gpos[1]), float(gpos[2])
    p.resetBasePositionAndOrientation(goal_sphere_id, [gx, gy, gz], [0,0,0,1])
    p.resetBasePositionAndOrientation(goal_ring_id,   [gx, gy, gz], [0,0,0,1])
    p.resetBasePositionAndOrientation(goal_post_id,   [gx, gy, gz-1.0], [0,0,0,1])


# ═══════════════════════════════════════════════════════════════════════════
#  SCENE BUILDER  ─  buildings / trees / rocks
# ═══════════════════════════════════════════════════════════════════════════
_BUILDING_COLORS = [
    [0.72, 0.70, 0.68, 1.0],   # light concrete
    [0.55, 0.52, 0.50, 1.0],   # dark concrete
    [0.62, 0.58, 0.50, 1.0],   # sandstone
    [0.42, 0.42, 0.46, 1.0],   # slate
    [0.68, 0.64, 0.56, 1.0],   # beige
]
_WINDOW_COLOR  = [0.45, 0.65, 0.82, 0.70]   # glass blue
_ROOF_COLOR    = [0.28, 0.28, 0.30, 1.0]
_TRUNK_COLOR   = [0.32, 0.20, 0.08, 1.0]
_CANOPY_COLORS = [
    [0.13, 0.52, 0.13, 0.88],
    [0.18, 0.60, 0.18, 0.88],
    [0.10, 0.44, 0.10, 0.88],
]
_ROCK_COLORS   = [
    [0.52, 0.48, 0.43, 1.0],
    [0.48, 0.45, 0.40, 1.0],
    [0.60, 0.55, 0.48, 1.0],
]

scene_bodies: list[int] = []


def build_scene():
    global scene_bodies
    for bid in scene_bodies:
        p.removeBody(bid)
    scene_bodies.clear()

    for idx, obs in enumerate(raw_env.obstacles):
        otype = obs["type"]
        cpos  = obs["pos"].tolist()

        # ── BUILDING ────────────────────────────────────────────────
        if otype == "box":
            he   = obs["he"].tolist()
            col  = _BUILDING_COLORS[idx % len(_BUILDING_COLORS)]
            w, h = he[0] * 2, he[2] * 2

            # Main structure
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=he, rgbaColor=col)
            col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=he)
            body = p.createMultiBody(0, col_shape, vis, cpos)
            scene_bodies.append(body)

            # Rooftop parapet
            roof_he = [he[0] + 0.05, he[1] + 0.05, 0.25]
            roof_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=roof_he,
                                           rgbaColor=_ROOF_COLOR)
            roof_pos = [cpos[0], cpos[1], cpos[2] + he[2] + 0.25]
            scene_bodies.append(p.createMultiBody(0, -1, roof_vis, roof_pos))

            # Rooftop HVAC box (visual variety)
            hvac_he = [he[0] * 0.25, he[1] * 0.25, 0.40]
            hvac_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=hvac_he,
                                           rgbaColor=_ROOF_COLOR)
            hvac_pos = [cpos[0] + he[0]*0.3, cpos[1] + he[1]*0.3,
                        cpos[2] + he[2] + 0.65]
            scene_bodies.append(p.createMultiBody(0, -1, hvac_vis, hvac_pos))

            # Window strips (2 per tall building face, visual only)
            if h > 6:
                win_he = [he[0] * 0.80, 0.05, he[2] * 0.05]
                for frac in [0.3, 0.65]:
                    wpos = [cpos[0], cpos[1] + he[1] + 0.06,
                            cpos[2] - he[2] + h * frac]
                    wv = p.createVisualShape(p.GEOM_BOX, halfExtents=win_he,
                                             rgbaColor=_WINDOW_COLOR)
                    scene_bodies.append(p.createMultiBody(0, -1, wv, wpos))

        # ── TREE ────────────────────────────────────────────────────
        elif otype == "cylinder":
            r, h  = obs["radius"], obs["height"]
            trunk_r = max(r * 0.30, 0.10)

            # Trunk
            t_vis = p.createVisualShape(p.GEOM_CYLINDER,
                                        radius=trunk_r, length=h,
                                        rgbaColor=_TRUNK_COLOR)
            t_col = p.createCollisionShape(p.GEOM_CYLINDER,
                                           radius=trunk_r, height=h)
            scene_bodies.append(p.createMultiBody(0, t_col, t_vis, cpos))

            # Canopy — 2 overlapping spheres for volume
            c_col = _CANOPY_COLORS[idx % len(_CANOPY_COLORS)]
            canopy_r = r * 1.8 + 0.4
            for dz, dr in [(0.35, 1.0), (0.60, 0.72)]:
                cv = p.createVisualShape(p.GEOM_SPHERE,
                                         radius=canopy_r * dr,
                                         rgbaColor=c_col)
                cp = [cpos[0], cpos[1], cpos[2] + h * dz]
                scene_bodies.append(p.createMultiBody(0, -1, cv, cp))

        # ── ROCK ────────────────────────────────────────────────────
        elif otype == "sphere":
            r   = obs["radius"]
            col = _ROCK_COLORS[idx % len(_ROCK_COLORS)]
            rv  = p.createVisualShape(p.GEOM_SPHERE, radius=r, rgbaColor=col)
            rc  = p.createCollisionShape(p.GEOM_SPHERE, radius=r)
            scene_bodies.append(p.createMultiBody(0, rc, rv, cpos))

            # Smaller accent sphere (gives craggy feel)
            r2  = r * 0.55
            rv2 = p.createVisualShape(p.GEOM_SPHERE, radius=r2,
                                       rgbaColor=[c + 0.06 for c in col[:3]] + [1.0])
            offset = [cpos[0] + r * 0.4, cpos[1] + r * 0.3, cpos[2] + r * 0.25]
            scene_bodies.append(p.createMultiBody(0, -1, rv2, offset))


# ═══════════════════════════════════════════════════════════════════════════
#  TRAIL  ─ cyan polyline behind the drone
# ═══════════════════════════════════════════════════════════════════════════
_trail_line_ids  : list[int]        = []
_trail_positions : list[np.ndarray] = []
TRAIL_MAX_PTS    = 100
TRAIL_INTERVAL   = 4   # add a point every N steps


def trail_add(pos: np.ndarray):
    _trail_positions.append(pos.copy())
    if len(_trail_positions) > TRAIL_MAX_PTS:
        p.removeUserDebugItem(_trail_line_ids.pop(0))
        _trail_positions.pop(0)
    if len(_trail_positions) >= 2:
        t = p.addUserDebugLine(
            _trail_positions[-2].tolist(),
            _trail_positions[-1].tolist(),
            lineColorRGB=[0.25, 0.70, 1.00],
            lineWidth=2.5,
        )
        _trail_line_ids.append(t)


def trail_clear():
    for tid in _trail_line_ids:
        p.removeUserDebugItem(tid)
    _trail_line_ids.clear()
    _trail_positions.clear()


# ═══════════════════════════════════════════════════════════════════════════
#  HUD
# ═══════════════════════════════════════════════════════════════════════════
_hud_ids: list[int] = []


def hud_update(episode: int, drone_pos: np.ndarray, goal_pos: np.ndarray,
               status: str = "", env=None):
    for tid in _hud_ids:
        p.removeUserDebugItem(tid)
    _hud_ids.clear()

    dist = float(np.linalg.norm(goal_pos - drone_pos))

    # Extract drone position for relative text placement
    x, y, z = drone_pos[0], drone_pos[1], drone_pos[2]

    # Fixed spatial offset pushing HUD towards the top-left of camera view
    # This acts like a fixed "mid-point" relative to the camera frame.
    ox, oy, oz = x - 5.0, y + 5.0, z + 12.0

    _hud_ids.append(p.addUserDebugText(
        f"[ EPISODE : {episode:03d} ]",
        [ox, oy, oz], textSize=2.0, textColorRGB=[0.2, 0.9, 1.0]
    ))
    _hud_ids.append(p.addUserDebugText(
        f"DISTANCE  : {dist:5.1f}m",
        [ox, oy, oz - 1.5], textSize=1.5, textColorRGB=[1.0, 0.8, 0.1]
    ))

    offset_z = oz - 3.0
    if env is not None:
        vel = float(np.linalg.norm(env.drone_vel))
        alt = float(env.drone_pos[2])
        roll, pitch, yaw = np.degrees(env.drone_rpy)
        thrust = float(getattr(env, 'last_thrust', 0.0))
        
        _hud_ids.append(p.addUserDebugText(
            f"ALTITUDE : {alt:5.1f}m",
            [ox, oy, offset_z], textSize=1.2, textColorRGB=[0.8, 0.8, 0.8]
        ))
        offset_z -= 1.2
        _hud_ids.append(p.addUserDebugText(
            f"VELOCITY : {vel:5.1f}m/s",
            [ox, oy, offset_z], textSize=1.2, textColorRGB=[0.8, 0.8, 0.8]
        ))
        offset_z -= 1.2
        _hud_ids.append(p.addUserDebugText(
            f"THRUST   : {thrust:5.1f}N",
            [ox, oy, offset_z], textSize=1.2, textColorRGB=[0.8, 0.8, 0.8]
        ))
        offset_z -= 1.2
        _hud_ids.append(p.addUserDebugText(
            f"R/P/Y    : {roll:4.0f} {pitch:4.0f} {yaw:4.0f}",
            [ox, oy, offset_z], textSize=1.2, textColorRGB=[0.8, 0.8, 0.8]
        ))
        offset_z -= 1.5

    if status:
        stat_col = [0.2, 1.0, 0.2] if "GOAL" in status else [1.0, 0.2, 0.2]
        _hud_ids.append(p.addUserDebugText(
            f"» {status} «",
            [ox, oy, offset_z], textSize=2.5, textColorRGB=stat_col,
            lifeTime=2.5
        ))


# ═══════════════════════════════════════════════════════════════════════════
#  CAMERA FOLLOW
# ═══════════════════════════════════════════════════════════════════════════
def cam_follow(pos: np.ndarray):
    p.resetDebugVisualizerCamera(
        cameraDistance      = 22,
        cameraYaw           = 40,
        cameraPitch         = -28,
        cameraTargetPosition = [float(pos[0]), float(pos[1]), float(pos[2])],
    )


# ═══════════════════════════════════════════════════════════════════════════
#  INITIAL SCENE
# ═══════════════════════════════════════════════════════════════════════════
build_drone()
update_goal(raw_env.goal)
build_scene()

def get_drone_orn(env):
    """Safely extracts the drone orientation from the raw environment."""
    if hasattr(env, 'drone_orn'):
        return list(env.drone_orn)
    elif hasattr(env, 'drone_rpy'):
        return p.getQuaternionFromEuler(env.drone_rpy)
    return [0, 0, 0, 1]  # Fallback to perfectly flat

# ═══════════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════
episode_count    = 1
rotor_angle      = 0.0
step_tick        = 0

# ── Frame-rate control ───────────────────────────────────────────────────
RENDER_FPS       = 60                       # visual refresh rate
RENDER_DT        = 1.0 / RENDER_FPS         # ≈ 0.0167 s
SIM_DT           = 0.075                    # simulation step interval (original pace)
HUD_INTERVAL     = 5                        # update HUD text every N render frames
CAM_INTERVAL     = 2                        # update camera every N render frames

print("[visualize] Running — press Ctrl-C or close window to quit.")
print(f"[visualize] Episode 1 starting …")

# Get initial state
drone_pos = raw_env.drone_pos.copy()
drone_orn = get_drone_orn(raw_env)
goal_pos  = raw_env.goal.copy()

# Setup smooth visual trackers
vis_pos = drone_pos.copy()
vis_orn = list(drone_orn)
done      = False

try:
    _sim_accum   = 0.0                      
    _prev_time   = time.perf_counter()
    _render_tick = 0                         

    while p.isConnected():
        _frame_start = time.perf_counter()
        _dt          = _frame_start - _prev_time
        _prev_time   = _frame_start

        if _dt > 0.25:
            _dt = 0.25

        _sim_accum += _dt

        # ── Step the model at the ORIGINAL pace (~13 Hz) ─────────────
        _terminal_pos = None        # position where the episode ended
        _terminal_orn = None
        while _sim_accum >= SIM_DT:
            _sim_accum -= SIM_DT

            # Save pre-step position (DummyVecEnv auto-resets after done)
            _pre_pos = raw_env.drone_pos.copy()
            _pre_orn = get_drone_orn(raw_env)

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_arr, info = viz_env.step(action)

            done = bool(done_arr[0])

            if done:
                # Save terminal position BEFORE auto-reset overwrites it
                _terminal_pos = _pre_pos
                _terminal_orn = _pre_orn
                _sim_accum = 0.0
                break

            # Update true physics targets (only if not done)
            drone_pos = raw_env.drone_pos.copy()
            drone_orn = get_drone_orn(raw_env)
            goal_pos  = raw_env.goal.copy()

            step_tick += 1

        # ── Smooth Interpolation for 60 FPS ──────────────────────────
        if not done:
            blend_speed = 0.35
            vis_pos = vis_pos + (drone_pos - vis_pos) * blend_speed
            vis_orn = p.getQuaternionSlerp(vis_orn, drone_orn, blend_speed)

        # ── Update visuals (every render frame) ──────────────────────
        rotor_angle = (rotor_angle + 22) % 360
        update_drone(vis_pos, vis_orn, rotor_angle)
        update_goal(goal_pos)

        if step_tick % TRAIL_INTERVAL == 0:
            trail_add(vis_pos)

        if _render_tick % HUD_INTERVAL == 0:
            hud_update(episode_count, vis_pos, goal_pos, env=raw_env)

        if _render_tick % CAM_INTERVAL == 0:
            cam_follow(vis_pos)

        p.stepSimulation()
        _render_tick += 1
        
        # ── Precise frame pacing ─────────────────────────────────────
        _elapsed = time.perf_counter() - _frame_start
        _sleep   = RENDER_DT - _elapsed
        if _sleep > 0:
            time.sleep(_sleep)

        # ── Episode end ──────────────────────────────────────────────
        if done:
            reason = info[0].get("reason", "")

            # Snap drone visually to the terminal (goal) position
            if _terminal_pos is not None:
                vis_pos = _terminal_pos.copy()
                vis_orn = list(_terminal_orn)
                update_drone(vis_pos, vis_orn, rotor_angle)

            if reason == "goal":
                print(f"[Ep {episode_count:>4}] ✓ GOAL REACHED  "
                      f"(steps: {step_tick})")
                hud_update(episode_count, vis_pos, goal_pos, "GOAL REACHED!", env=raw_env)

                # Drone flashes green three times
                for _ in range(3):
                    for bid in ["body", "arm_0", "arm_1"]:
                        p.changeVisualShape(drone_parts[bid], -1,
                                            rgbaColor=[0.1, 0.9, 0.2, 1.0])
                    time.sleep(0.15)
                    for bid in ["body", "arm_0", "arm_1"]:
                        p.changeVisualShape(drone_parts[bid], -1,
                                            rgbaColor=DARK if "arm" not in bid else MID)
                    time.sleep(0.15)

                # Wait at goal so the user can see the drone there
                time.sleep(2.0)

            else:
                label = "COLLISION" if reason == "collision" else "TIMEOUT"
                print(f"[Ep {episode_count:>4}] ✗ {label:<10}  "
                      f"(steps: {step_tick})")
                hud_update(episode_count, vis_pos, goal_pos, label, env=raw_env)

                # Drone flashes red
                for bid in ["body", "arm_0", "arm_1"]:
                    p.changeVisualShape(drone_parts[bid], -1,
                                        rgbaColor=[0.9, 0.1, 0.1, 1.0])
                time.sleep(0.6)
                for bid in ["body", "arm_0", "arm_1"]:
                    p.changeVisualShape(drone_parts[bid], -1,
                                        rgbaColor=DARK if "arm" not in bid else MID)
                time.sleep(1.0)

            # ── Move drone to new start position ─────────────────────
            episode_count += 1
            step_tick = 0
            trail_clear()

            # Snap visual trackers to the already-reset start position
            drone_pos = raw_env.drone_pos.copy()
            drone_orn = get_drone_orn(raw_env)
            vis_pos   = drone_pos.copy()
            vis_orn   = list(drone_orn)
            goal_pos  = raw_env.goal.copy()
            done      = False
            _sim_accum = 0.0

            update_drone(vis_pos, vis_orn, rotor_angle)
            update_goal(goal_pos)
            build_scene()
            cam_follow(vis_pos)
            hud_update(episode_count, vis_pos, goal_pos, env=raw_env)

            # Brief pause at start so user can see the new position
            time.sleep(1.5)
            _prev_time = time.perf_counter()

            print(f"[visualize] Episode {episode_count} starting …")

except (KeyboardInterrupt, p.error):
    pass
finally:
    print("[visualize] Disconnecting …")
    try:
        p.disconnect()
    except Exception:
        pass
    sys.exit(0)
print("Script finished")