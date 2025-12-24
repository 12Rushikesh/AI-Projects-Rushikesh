import os

# Root project directory
PROJECT_NAME = "damage-ai-agent"

# Folder & file structure
structure = {
    "app": [
        "__init__.py",
        "main.py",
        "model.py",
        "agent.py",
        "feedback.py",
        "analytics.py",
        "llm_reasoning.py",
    ],
    "data": {
        "incoming": [],
        "feedback": {
            "dent": [],
            "hole": [],
            "rust": [],
            "not_damaged": [],
        },
        "feedback_meta": [],
        "yolo_labels": [],
        "errors": [],
    },
    "retraining": [
        "prepare_dataset.py",
        "retrain.py",
        "data.yaml",
    ],
    "models": [
        "best.pt",
    ],
    "requirements.txt": None,
    "README.md": None,
}


def create_structure(base_path, tree):
    """
    Recursively create folders and files
    """
    for name, content in tree.items():
        path = os.path.join(base_path, name)

        # Case 1: File at root
        if content is None:
            open(path, "w").close()

        # Case 2: Folder with list of files
        elif isinstance(content, list):
            os.makedirs(path, exist_ok=True)
            for file in content:
                open(os.path.join(path, file), "w").close()

        # Case 3: Folder with nested folders
        elif isinstance(content, dict):
            os.makedirs(path, exist_ok=True)
            create_structure(path, content)


def main():
    base_path = os.path.abspath(PROJECT_NAME)
    os.makedirs(base_path, exist_ok=True)

    create_structure(base_path, structure)

    print(f"✅ Project structure created at:\n{base_path}")


if __name__ == "__main__":
    main()
