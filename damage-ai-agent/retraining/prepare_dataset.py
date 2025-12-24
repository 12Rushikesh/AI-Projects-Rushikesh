from pathlib import Path
import shutil

FB = Path("D:/Rushikesh/project/AI Agent/damage-ai-agent/data/feedback/feedback")
OUT = Path("D:/Rushikesh/project/AI Agent/damage-ai-agent/data/dataset/images/train")
OUT.mkdir(parents=True, exist_ok=True)

for cls in ["dent","hole","rust"]:
    for img in (FB/cls).glob("*.*"):
        shutil.copy(img, OUT/img.name)

print("Dataset ready")
