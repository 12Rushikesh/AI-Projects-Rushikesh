# app/rl_trainer.py
"""
Offline PPO trainer that uses logged experience (data/rl_experience.jsonl)
to train a lightweight discrete policy controlling agent actions.

Usage:
    python -m app.rl_trainer

Requirements:
    pip install stable-baselines3[extra] gym numpy
"""
import os
import json
import numpy as np

# Try to import stable-baselines3
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    import gym
    from gym import spaces
except Exception as e:
    raise ImportError(
        "Please install stable-baselines3 and gym to use rl_trainer:\n"
        "pip install stable-baselines3[extra] gym numpy"
    ) from e

from app.rl_memory import read_all

# Define mapping of actions to indices
ACTION_MAP = {
    "AUTO_ACCEPT": 0,
    "ASK_HUMAN": 1,
    "REJECT": 2
}
INV_ACTION_MAP = {v: k for k, v in ACTION_MAP.items()}

# CLASSES (must match your app)
CLASSES = ["dent", "hole", "rust", "not_damaged"]

def build_feature_vector(record):
    """
    Convert a logged record into a fixed-size numeric observation vector.
    The format (example):
      - one-hot for predicted damage_type (len(CLASSES))
      - yolo_confidence (0..1)
      - vision_confidence (0..1) mapped from "low/medium/high"
      - action index (one-hot of last action) OPTIONAL (we exclude)
    Returns np.float32 array.
    """
    st = record.get("state", {})
    # yolo preds
    yolo = st.get("yolo", {})
    # The code expects simple keys:
    # yolo may be dict like {"label":"dent","confidence":0.8} or complex
    yolo_label = yolo.get("label", None) or st.get("yolo_label", None) or "not_damaged"
    try:
        label_idx = CLASSES.index(yolo_label) if yolo_label in CLASSES else len(CLASSES) - 1
    except Exception:
        label_idx = len(CLASSES) - 1

    one_hot_label = np.zeros(len(CLASSES), dtype=np.float32)
    if 0 <= label_idx < len(CLASSES):
        one_hot_label[label_idx] = 1.0

    # yolo numeric confidence
    yolo_conf = yolo.get("confidence", None)
    if yolo_conf is None:
        yolo_conf = float(st.get("yolo_confidence", 0.0))
    try:
        yolo_conf = float(yolo_conf)
    except Exception:
        yolo_conf = 0.0
    yolo_conf = np.clip(yolo_conf, 0.0, 1.0)

    # vision confidence from agent state (low/medium/high)
    vl = st.get("agent", {}) or st.get("vision", {})
    vl_conf = vl.get("confidence", st.get("vision_confidence", "low"))
    mapping = {"low": 0.0, "medium": 0.5, "high": 1.0}
    try:
        vl_conf_f = mapping.get(vl_conf, float(vl_conf) if isinstance(vl_conf, (int,float)) else 0.0)
    except Exception:
        vl_conf_f = 0.0

    vec = np.concatenate([one_hot_label, np.array([yolo_conf, vl_conf_f], dtype=np.float32)])
    return vec

class LoggedExperienceEnv(gym.Env):
    """
    A Gym environment that yields logged experiences as one-step episodes.
    Observations: fixed vector extracted from a record
    Actions: Discrete(3) -> 0:AUTO_ACCEPT,1:ASK_HUMAN,2:REJECT
    Each episode lasts exactly one step; reward is the logged reward.
    """

    def __init__(self, records):
        super().__init__()
        self.records = records or []
        self.current = None
        # observation size:
        obs_example = build_feature_vector(self.records[0]) if len(self.records) > 0 else np.zeros(len(CLASSES) + 2, dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=obs_example.shape, dtype=np.float32)
        self.action_space = spaces.Discrete(len(ACTION_MAP))
        self._idx = 0

    def reset(self):
        if not self.records:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        # sample a random record for variety
        self._idx = np.random.randint(0, len(self.records))
        self.current = self.records[self._idx]
        obs = build_feature_vector(self.current)
        return obs

    def step(self, action):
        # reward uses logged reward (note: offline training)
        if not self.current:
            return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, True, {}
        reward = float(self.current.get("reward", 0.0))
        done = True
        info = {"logged_action": self.current.get("action")}
        # next state: zeros (episode length 1)
        next_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return next_obs, reward, done, info

def load_records_to_env():
    recs = read_all()
    # Filter records with usable state
    recs = [r for r in recs if r.get("state") is not None]
    if not recs:
        raise ValueError("No RL records found in data/rl_experience.jsonl")
    return LoggedExperienceEnv(recs)

def train(total_timesteps=10000, model_out="models/decision_policy.zip"):
    env = load_records_to_env()
    # wrapper for SB3
    vec_env = DummyVecEnv([lambda: env])
    model = PPO("MlpPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    model.save(model_out)
    print(f"Saved policy to {model_out}")
    return model

def evaluate(model_path="models/decision_policy.zip", n_eval=200):
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)
    env = load_records_to_env()
    vec_env = DummyVecEnv([lambda: env])
    model = PPO.load(model_path, env=vec_env)
    recs = read_all()
    if not recs:
        print("No records to evaluate.")
        return
    total_reward = 0.0
    for _ in range(min(n_eval, len(recs))):
        obs = vec_env.reset()
        action, _ = model.predict(obs, deterministic=True)
        _, reward, done, info = vec_env.step(action)
        total_reward += float(reward[0])
    avg = total_reward / max(1, min(n_eval, len(recs)))
    print(f"Avg reward on logged data: {avg:.4f}")
    return avg

if __name__ == "__main__":
    # Simple CLI trainer
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=10000)
    p.add_argument("--out", type=str, default="models/decision_policy.zip")
    args = p.parse_args()
    model = train(total_timesteps=args.timesteps, model_out=args.out)
    evaluate(args.out)
