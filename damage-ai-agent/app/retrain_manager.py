from pathlib import Path
import subprocess

DATASET_IMG = Path(r"D:\Rushikesh\project\AI Agent\damage-ai-agent\data\dataset\images\train")

def should_retrain(threshold=500):
    return len(list(DATASET_IMG.glob("*.*"))) >= threshold

def trigger_retrain():
    subprocess.Popen(["python", "retraining/retrain.py"])
