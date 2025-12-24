# retraining/retrain.py
from pathlib import Path
import subprocess
import yaml
import sys
from pathlib import Path

PROJECT_ROOT = Path(r"D:/Rushikesh/project/AI Agent/damage-ai-agent")
sys.path.append(str(PROJECT_ROOT))


# --------------------------------------------------
# ABSOLUTE PROJECT ROOT
# --------------------------------------------------
PROJECT_ROOT = Path(r"D:/Rushikesh/project/AI Agent/damage-ai-agent")

DATASET_DIR = PROJECT_ROOT / "data/dataset"
RUNS_DIR = PROJECT_ROOT / "runs/detect/train/weights"
MODEL_DIR = PROJECT_ROOT / "models"

DATA_YAML = PROJECT_ROOT / "retraining/data.yaml"

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
CLASSES = ["dent", "hole", "rust"]

EPOCHS = 100
IMG_SIZE = 640
BATCH = 8          # laptop safe
DEVICE = "0"       # change to 0 if GPU later

# --------------------------------------------------
# CREATE data.yaml (ALWAYS SAFE)
# --------------------------------------------------
def create_data_yaml():
    data = {
        "path": str(DATASET_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(CLASSES)}
    }

    with open(DATA_YAML, "w") as f:
        yaml.dump(data, f)

    print(f"✅ data.yaml created at {DATA_YAML}")

# --------------------------------------------------
# GET BASE MODEL (TRUE FINE-TUNING)
# --------------------------------------------------
def get_base_model():
    best_model = MODEL_DIR / "best.pt"

    if best_model.exists():
        print("🔁 Fine-tuning from existing best.pt")
        return str(best_model)

    print("🆕 No fine-tuned model found → using yolov8n.pt")
    return "yolov8n.pt"

# --------------------------------------------------
# TRAIN
# --------------------------------------------------
def train():
    model_path = get_base_model()

    cmd = [
        "yolo", "train",
        f"model={model_path}",
        f"data={DATA_YAML}",
        f"epochs={EPOCHS}",
        f"imgsz={IMG_SIZE}",
        f"batch={BATCH}",
        f"device={DEVICE}",
        f"project={PROJECT_ROOT / 'runs/detect'}",
        "name=train",
        "exist_ok=True"
    ]

    print("🚀 Starting YOLO retraining...")
    subprocess.run(cmd, check=True)
    print("✅ Training completed")

# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":
    from retraining.split_data import split_train_val

    print("🔀 Splitting train / val...")
    split_train_val()

    print("📝 Creating data.yaml...")
    create_data_yaml()

    train()
