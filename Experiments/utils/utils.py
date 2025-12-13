import os
import json
from typing import Dict


def load_processed_ids(CHECKPOINT_FILE: str) -> Dict:
    if not os.path.exists(CHECKPOINT_FILE):
        return set()
    with open(CHECKPOINT_FILE, "r") as f:
        return set(line.strip() for line in f)


def save_success(
    result_dict: Dict, prompt_id: int, OUTPUT_FILE: str, CHECKPOINT_FILE: str
) -> None:
    """Only called when we actually get data."""
    # Append data
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(result_dict) + "\n")

    # Update Checkpoint
    with open(CHECKPOINT_FILE, "a", encoding="utf-8") as f:
        f.write(f"{prompt_id}\n")
