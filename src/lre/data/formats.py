"""
Turn records from `lre.data.load` into the files COCONUT and CODI train and evaluate on.

COCONUT (third_party/coconut/dataset.py) reads a JSON list of
    {"question": ..., "steps": [...], "answer": ...}

CODI (third_party/codi/train.py) reads GSM8k-Aug's own fields,
    {"question": ..., "cot": "<<...>> <<...>>", "answer": ...}
from Hugging Face. To train it on our files, load them with
    load_dataset("json", data_files=path)["train"]
in place of load_dataset("zen-E/GSM8k-Aug")["train"].
"""

import json
from pathlib import Path


def to_coconut(example: dict) -> dict:
    """The record in COCONUT's format. Other fields (id, rating, ...) are dropped."""
    return {
        "question": example["question"],
        "steps": example["steps"],
        "answer": example["answer"],
    }


def to_codi(example: dict) -> dict:
    """The record in CODI's (GSM8k-Aug's) format: the steps joined back into one string."""
    return {
        "question": example["question"],
        "cot": " ".join(example["steps"]),
        "answer": example["answer"],
    }


def write_json(records: list[dict], path: str | Path) -> None:
    """Write `records` to `path` as one JSON list, making any missing folders."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(records, f)
