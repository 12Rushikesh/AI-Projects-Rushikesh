# rl/ppo_train.py
import os
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from rl.replay_env import ReplayStepEnv

MODEL_DIR = Path(r"D:\Rushikesh\project\AI Agent\damage-ai-agent\data/rl/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "ppo_policy.zip"

def train(total_timesteps=20000):
    env = DummyVecEnv([lambda: ReplayStepEnv()])
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    model.save(str(MODEL_PATH))
    print(f"Saved PPO policy to {MODEL_PATH}")

if __name__ == "__main__":
    # simple CLI
    train(total_timesteps=int(os.environ.get("PPO_TIMESTEPS", 20000)))
