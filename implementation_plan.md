# Phase 2: Mid-Level Dynamic Drone Control

Rewrite [drone_env.py](file:///d:/sem6/RL/DroneNavigation/RL-Drone-Navigation/drone_env.py) from kinematic velocity control to physics-based dynamic control with a PD attitude controller, gravity, and stability-aware reward shaping.

## Proposed Changes

### [MODIFY] [drone_env.py](file:///d:/sem6/RL/DroneNavigation/RL-Drone-Navigation/drone_env.py)

The entire environment transitions from "hoverboard" to physically realistic dynamics. Here is a summary of every change:

---

#### 1. New Action Space (3 → 4 dimensions)

| Phase 1 | Phase 2 |
|---------|---------|
| `[vx, vy, vz]` — velocity command | `[thrust, roll_cmd, pitch_cmd, yaw_rate_cmd]` |

- **Thrust** `∈ [-1, 1]` → mapped to `[0, 2g]` so that 0.0 = hover thrust (1g) and +1.0 = double gravity
- **Roll / Pitch** `∈ [-1, 1]` → mapped to `[-MAX_ANGLE, +MAX_ANGLE]` (≈ 30°)
- **Yaw Rate** `∈ [-1, 1]` → mapped to `[-MAX_YAW_RATE, +MAX_YAW_RATE]` (≈ 90°/s)

---

#### 2. New State Variables

```python
self.drone_rpy      = np.zeros(3)   # [roll, pitch, yaw] in radians
self.drone_ang_vel  = np.zeros(3)   # angular velocity [p, q, r] rad/s
self.drone_vel      = np.zeros(3)   # linear velocity in world frame
```

---

#### 3. Physics in [step()](file:///d:/sem6/RL/DroneNavigation/RL-Drone-Navigation/drone_env.py#283-328) — PD Attitude Controller + Force Integration

The [step()](file:///d:/sem6/RL/DroneNavigation/RL-Drone-Navigation/drone_env.py#283-328) function will:

1. **Decode the action** into desired thrust, desired roll/pitch angles, and desired yaw rate.
2. **PD Attitude Control Loop** — compute torques:
   - `roll_torque  = Kp_rp * (desired_roll  − current_roll)  − Kd_rp * current_p`
   - `pitch_torque = Kp_rp * (desired_pitch − current_pitch) − Kd_rp * current_q`
   - `yaw_torque   = Kp_yaw * (desired_yaw_rate − current_r)`
3. **Apply angular acceleration** → update `drone_ang_vel` and `drone_rpy`.
4. **Apply thrust** — thrust acts along the drone's local Z-axis (body-up):
   - Convert thrust to world frame using the current orientation rotation matrix.
   - Add gravity `[0, 0, -9.81]`.
   - Integrate to get new velocity and position.
5. **Clamp** roll/pitch/yaw, angular velocities, linear velocity to safe ranges.

This stays fully in NumPy (no PyBullet physics body needed in the env itself — PyBullet is only used visually), keeping training fast with vectorized envs.

---

#### 4. Expanded Observation Space (20 → 26 dimensions)

| Slot | Dims | Content |
|------|------|---------|
| 0–2 | 3 | Goal direction (unit vector) |
| 3 | 1 | Goal distance (normalised) |
| 4–6 | 3 | Position (normalised by world dims) |
| 7–9 | 3 | Linear velocity (normalised) |
| **10–12** | **3** | **Roll, Pitch, Yaw (normalised by π)** ← NEW |
| **13–15** | **3** | **Angular velocity (normalised)** ← NEW |
| 16–25 | 10 | LiDAR readings (unchanged) |

---

#### 5. New Reward Function

The base navigation rewards remain, with new stability terms added:

```
reward  = (prev_dist − new_dist) × 3.0           # progress toward goal
reward -= 0.05                                     # time penalty
reward -= boundary_penalty (if clipped)
reward -= obstacle_proximity_penalty

# NEW stability penalties:
reward -= 2.0 × (|roll|/MAX_ANGLE + |pitch|/MAX_ANGLE)  # penalise tilt
reward -= 0.5 × |angular_vel| / MAX_ANG_VEL             # penalise spin
reward -= 5.0  if |roll| or |pitch| > 60°               # extreme tilt kill
```

On crash / extreme flip: `reward = -100, done = True`.

---

#### 6. New Constants

```python
MASS        = 1.0      # kg
GRAVITY     = 9.81     # m/s²
DT          = 0.05     # simulation timestep (s)
MAX_ANGLE   = 0.5236   # 30° in radians
MAX_YAW_RATE = 1.5708  # 90°/s
MAX_ANG_VEL = 3.0      # rad/s clamp
MAX_VEL     = 5.0      # m/s
KP_RP       = 15.0     # roll/pitch proportional gain
KD_RP       = 5.0      # roll/pitch derivative gain
KP_YAW      = 5.0      # yaw proportional gain
```

---

#### 7. Visualization Compatibility

The [visualize.py](file:///d:/sem6/RL/DroneNavigation/RL-Drone-Navigation/visualize.py) file already has a [get_drone_orn()](file:///d:/sem6/RL/DroneNavigation/RL-Drone-Navigation/visualize.py#431-438) helper that checks for [drone_orn](file:///d:/sem6/RL/DroneNavigation/RL-Drone-Navigation/visualize.py#431-438) or `drone_rpy` attributes. Our new env exposes `drone_rpy`, so visualization will automatically pick up the drone's tilt via `p.getQuaternionFromEuler(env.drone_rpy)`. **No changes needed to [visualize.py](file:///d:/sem6/RL/DroneNavigation/RL-Drone-Navigation/visualize.py).**

---

## Verification Plan

### Automated Tests

Since there are no existing tests in the project, I will verify with a short smoke-test script:

```
python -c "from drone_env import DroneEnv; e = DroneEnv(); o, _ = e.reset(); assert o.shape == (26,); o2, r, d, t, i = e.step(e.action_space.sample()); assert o2.shape == (26,); print('PASS')"
```

This confirms:
1. The env instantiates without error
2. [reset()](file:///d:/sem6/RL/DroneNavigation/RL-Drone-Navigation/drone_env.py#63-91) returns a 26-dim observation
3. [step()](file:///d:/sem6/RL/DroneNavigation/RL-Drone-Navigation/drone_env.py#283-328) with a random action returns correctly shaped output
4. No exceptions are raised

### Manual Verification

Since this is Phase 2 of an RL project and the old trained model won't work with the new action/obs spaces, manual verification consists of:

1. **Run the smoke test above** — confirms env is structurally valid
2. **Inspect the new code** for correctness of the PD controller math and reward function
3. **After retraining** (outside this task's scope), run [visualize.py](file:///d:/sem6/RL/DroneNavigation/RL-Drone-Navigation/visualize.py) to observe the drone tilting/stabilising with the new physics
