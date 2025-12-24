# rl/replay_env.py
import json
from pathlib import Path
import numpy as np
import gymnasium as gym
from gymnasium import spaces

RL_LOG = Path(r"D:\Rushikesh\project\AI Agent\damage-ai-agent\data/rl/rl_steps.jsonl")

# map actions to discrete indices
ACTION_MAP = {
    "AUTO_ACCEPT": 0,
    "PREVENTIVE_MAINTENANCE": 1,
    "ASK_HUMAN": 2,
    "OTHER": 3
}
NUM_ACTIONS = len(ACTION_MAP)

def _action_to_index(a):
    return ACTION_MAP.get(a, ACTION_MAP["OTHER"])

class ReplayStepEnv(gym.Env):
    """
    Minimal Gym env that yields one logged RL step as a single-step episode.
    Observation: vector of floats (features). Action: discrete index.
    Reward: from logged reward.
    This is a simple offline wrapper to let PPO learn from logged data.
    """
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        # we assume the state vector has these fixed fields:
        # conf_dent, conf_hole, conf_rust, conf_not_damaged, mean_conf, num_boxes
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(6,), dtype=float)
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self._load_data()
        self._idx = 0

    def _load_data(self):
        self._steps = []
        if not RL_LOG.exists():
            self._steps = []
            return
        with open(RL_LOG, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    s = r.get("state", {})
                    y = s.get("yolo_summary", {})
                    num_boxes = float(s.get("num_boxes", 0))
                    obs = [
                        float(y.get("conf_dent", 0.0)),
                        float(y.get("conf_hole", 0.0)),
                        float(y.get("conf_rust", 0.0)),
                        float(y.get("conf_not_damaged", 0.0)),
                        float(y.get("mean_conf", 0.0)),
                        min(1.0, num_boxes / 10.0)  # normalize box count to [0,1] with cap at 10
                    ]
                    act = _action_to_index(r.get("action", "OTHER"))
                    rew = float(r.get("reward", 0.0))
                    self._steps.append({"obs": np.array(obs, dtype=float), "act": act, "rew": rew})
                except Exception:
                    continue

    def reset(self, *, seed=None, options=None):
        # sequential replay; if index beyond end, loop
        if not self._steps:
            # empty dataset: return zero observation
            self._idx = 0
            return np.zeros(self.observation_space.shape, dtype=float), {}
        self._idx = (self._idx) % len(self._steps)
        obs = self._steps[self._idx]["obs"]
        return obs, {}

    def step(self, action):
        # give stored reward and done True for one-step episode
        if not self._steps:
            return np.zeros(self.observation_space.shape, dtype=float), 0.0, True, False, {}
        item = self._steps[self._idx]
        reward = float(item["rew"])
        done = True
        info = {"logged_action": item["act"]}
        # move index for next reset
        self._idx = (self._idx + 1) % len(self._steps)
        # observation for next reset (not used)
        next_obs = self._steps[self._idx]["obs"]
        return next_obs, reward, done, False, info
