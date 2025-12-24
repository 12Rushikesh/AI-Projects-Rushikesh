# retraining/split_data.py
import random
import shutil
from pathlib import Path

TRAIN_RATIO = 0.8

BASE = Path("D:/Rushikesh/project/AI Agent/damage-ai-agent/data/dataset")
IMG_DIR = BASE / "images"
LBL_DIR = BASE / "labels"

def split_train_val():
    train_imgs = list((IMG_DIR / "train").glob("*.jpg"))
    random.shuffle(train_imgs)

    val_count = int(len(train_imgs) * (1 - TRAIN_RATIO))
    val_imgs = train_imgs[:val_count]

    for img_path in val_imgs:
        label_path = (LBL_DIR / "train" / f"{img_path.stem}.txt")

        shutil.move(str(img_path), IMG_DIR / "val" / img_path.name)

        if label_path.exists():
            shutil.move(str(label_path), LBL_DIR / "val" / label_path.name)

    print(f"✅ Split completed: {len(val_imgs)} moved to val")

if __name__ == "__main__":
    split_train_val()
