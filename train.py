"""
train.py — PPO Training for Drone Navigation
=============================================
• 8 parallel envs  (DummyVecEnv)
• VecNormalize     (obs + reward)
• 2 000 000 timesteps
• EvalCallback     (saves best model automatically)
• Rolling-mean reward curve
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import warnings

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
# from stable_baselines3.common.utils import sync_envs_normalization

from drone_env import DroneEnv

warnings.filterwarnings("ignore")

# ── Hyper-parameters ─────────────────────────────────────────────────────────
N_ENVS        = 8
TOTAL_STEPS   = 2_000_000
EVAL_FREQ     = 50_000        # steps between evaluations (per env)
N_EVAL_EPS    = 30
ROLLING_WIN   = 100           # episodes in rolling average

os.makedirs("models", exist_ok=True)
os.makedirs("logs",   exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
#  CALLBACK — episode reward recorder with smooth plot
# ═══════════════════════════════════════════════════════════════════════════
class EpisodeRewardCallback(BaseCallback):
    """Accumulates per-episode rewards from env[0] and plots a rolling mean."""

    def __init__(self, rolling_window: int = 100):
        super().__init__()
        self.ep_rewards  : list[float] = []
        self._ep_running : float       = 0.0
        self.rolling_window            = rolling_window

    def _on_step(self) -> bool:
        self._ep_running += float(self.locals["rewards"][0])
        if self.locals["dones"][0]:
            self.ep_rewards.append(self._ep_running)
            self._ep_running = 0.0
        return True

    def plot(self, save_path: str = "training_curve.png"):
        rewards  = np.array(self.ep_rewards, dtype=np.float32)
        if len(rewards) < self.rolling_window:
            print("[Warning] Too few episodes to plot a rolling average.")
            return
        kernel   = np.ones(self.rolling_window) / self.rolling_window
        rolling  = np.convolve(rewards, kernel, mode="valid")
        episodes = np.arange(len(rolling)) + self.rolling_window

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(rewards, alpha=0.25, color="#4a9edd", linewidth=0.8,
                label="Episode reward")
        ax.plot(episodes, rolling, color="#4a9edd", linewidth=2.0,
                label=f"Rolling mean (n={self.rolling_window})")
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Episode", fontsize=12)
        ax.set_ylabel("Total reward", fontsize=12)
        ax.set_title("PPO — Drone Navigation Training Curve", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"[train] Curve saved → {save_path}")


# ═══════════════════════════════════════════════════════════════════════════
#  ENVIRONMENTS
# ═══════════════════════════════════════════════════════════════════════════
print("[train] Building environments …")

train_env = make_vec_env(DroneEnv, n_envs=N_ENVS)
train_env = VecNormalize(
    train_env,
    norm_obs=True,
    norm_reward=True,
    clip_obs=10.0,
)

# Separate eval env — VecNormalize stats are synced automatically
# by EvalCallback before every evaluation run
eval_env = make_vec_env(DroneEnv, n_envs=1)
eval_env = VecNormalize(
    eval_env,
    norm_obs=True,
    norm_reward=False,    # don't normalise rewards during evaluation
    clip_obs=10.0,
    training=False,
)

# ═══════════════════════════════════════════════════════════════════════════
#  MODEL
# ═══════════════════════════════════════════════════════════════════════════
print("[train] Building PPO model …")

model = PPO(
    "MlpPolicy",
    train_env,
    verbose=1,
    device="cpu",
    # --- PPO core ---
    n_steps     = 2048,
    batch_size  = 256,
    n_epochs    = 10,
    gamma       = 0.995,
    gae_lambda  = 0.95,
    clip_range  = 0.20,
    # --- exploration ---
    ent_coef    = 0.005,
    # --- learning rate ---
    learning_rate = 3e-4,
    # --- network: 3 hidden layers, wider than default ---
    policy_kwargs = dict(net_arch=[256, 256, 128]),
)

# ═══════════════════════════════════════════════════════════════════════════
#  CALLBACKS
# ═══════════════════════════════════════════════════════════════════════════
reward_cb = EpisodeRewardCallback(rolling_window=ROLLING_WIN)

eval_cb = EvalCallback(
    eval_env,
    best_model_save_path = "./models/",
    log_path             = "./logs/",
    eval_freq            = max(EVAL_FREQ // N_ENVS, 1),
    n_eval_episodes      = N_EVAL_EPS,
    deterministic        = True,
    verbose              = 1,
)

# ═══════════════════════════════════════════════════════════════════════════
#  TRAIN
# ═══════════════════════════════════════════════════════════════════════════
print(f"[train] Starting — {TOTAL_STEPS:,} timesteps across {N_ENVS} envs …\n")

model.learn(
    total_timesteps = TOTAL_STEPS,
    callback        = [reward_cb, eval_cb],
    progress_bar    = True,
)

# ═══════════════════════════════════════════════════════════════════════════
#  SAVE
# ═══════════════════════════════════════════════════════════════════════════
model.save("drone_model")
train_env.save("vec_normalize.pkl")
reward_cb.plot("training_curve.png")

print("\n[train] Done.")
print("  Saved: drone_model.zip")
print("  Saved: vec_normalize.pkl")
print("  Saved: training_curve.png")
print("  Best model checkpoint: models/best_model.zip")
