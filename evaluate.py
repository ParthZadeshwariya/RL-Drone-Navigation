"""
evaluate.py — Comprehensive Evaluation of the Trained Drone Model
==================================================================
Loads the saved VecNormalize stats and runs 200 deterministic episodes,
reporting success rate, collision rate, timeout rate, and step statistics.
"""

import numpy as np
import warnings

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from drone_env import DroneEnv

warnings.filterwarnings("ignore")

EPISODES     = 200
MODEL_PATH   = "drone_model"          # or "models/best_model"
VN_PATH      = "vec_normalize.pkl"

# ── Load environment + normalization stats ────────────────────────────────
env = DummyVecEnv([DroneEnv])
env = VecNormalize.load(VN_PATH, env)
env.training   = False     # keep normalization stats frozen
env.norm_reward = False    # never normalise rewards during evaluation

model = PPO.load(MODEL_PATH, env=env)

# ── Track outcomes ────────────────────────────────────────────────────────
results = {
    "success":   0,
    "collision": 0,
    "timeout":   0,
    "other":     0,
}
steps_to_goal    : list[int]   = []
rewards_per_ep   : list[float] = []
episode_ep_reward: float       = 0.0

obs = env.reset()

for ep in range(EPISODES):
    obs = env.reset()
    episode_ep_reward = 0.0
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done_arr, info = env.step(action)
        episode_ep_reward += float(reward[0])
        done = bool(done_arr[0])

        if done:
            reason = info[0].get("reason", "other")
            if reason == "goal":
                results["success"] += 1
                steps_to_goal.append(env.envs[0].steps)
            elif reason == "collision":
                results["collision"] += 1
            elif reason == "timeout":
                results["timeout"] += 1
            else:
                results["other"] += 1
            rewards_per_ep.append(episode_ep_reward)

    if (ep + 1) % 20 == 0:
        so_far = results["success"]
        print(f"  [{ep+1:>3}/{EPISODES}]  success so far: "
              f"{so_far}/{ep+1}  ({so_far/(ep+1)*100:.1f}%)")

# ── Report ────────────────────────────────────────────────────────────────
n = EPISODES
sep = "─" * 44
print(f"\n{sep}")
print(f"  {'EVALUATION RESULTS':^40}")
print(sep)
print(f"  Episodes evaluated   : {n}")
print(f"  Success              : {results['success']:>4}  "
      f"({results['success']/n*100:5.1f} %)")
print(f"  Collision            : {results['collision']:>4}  "
      f"({results['collision']/n*100:5.1f} %)")
print(f"  Timeout              : {results['timeout']:>4}  "
      f"({results['timeout']/n*100:5.1f} %)")
if results["other"]:
    print(f"  Other                : {results['other']:>4}  "
          f"({results['other']/n*100:5.1f} %)")
print(sep)

if steps_to_goal:
    arr = np.array(steps_to_goal)
    print(f"  Steps to goal (mean) : {arr.mean():.1f}")
    print(f"  Steps to goal (med)  : {np.median(arr):.1f}")
    print(f"  Steps to goal (min)  : {arr.min()}")
    print(f"  Steps to goal (max)  : {arr.max()}")
    print(sep)

if rewards_per_ep:
    r = np.array(rewards_per_ep)
    print(f"  Ep reward  mean      : {r.mean():.1f}")
    print(f"  Ep reward  std       : {r.std():.1f}")
    print(f"  Ep reward  min/max   : {r.min():.1f} / {r.max():.1f}")
    print(sep)

print()
