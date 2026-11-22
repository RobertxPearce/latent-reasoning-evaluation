"""
Load the datasets from Hugging Face into one shared record shape:

    {"id": "aug-train-0", "question": "...", "steps": ["<<600*30/100=180>>", ...], "answer": "360"}

Every other function in `lre.data` takes a list of these records.
"""

import re

from datasets import load_dataset

from lre.metrics import extract_answer

# A calculation step in GSM8K notation, e.g. "<<600*30/100=180>>"
STEP = re.compile(r"<<[^>]*>>")


def load_gsm8k_aug(split: str) -> list[dict]:
    """Load GSM8k-Aug ("train": 385,620 problems, "test": 1,319), the data COCONUT and
    CODI train on.

    `steps` is the `cot` field split on spaces, as COCONUT's own preprocessing does. A
    few problems have an empty `cot`; they get `steps == []` (zero steps), not [""].
    `answer` is kept as given. A few training answers aren't plain numbers (e.g. "2/5").
    """
    ds = load_dataset("zen-E/GSM8k-Aug", split=split)

    records = []
    for i, row in enumerate(ds):
        records.append({
            "id": f"aug-{split}-{i}",
            "question": row["question"],
            # split() rather than split(" "), so an empty cot gives []
            "steps": row["cot"].split(),
            "answer": row["answer"],
        })

    return records


def load_e2h_gsm8k() -> list[dict]:
    """Load E2H-GSM8K: the 1,319 GSM8K test problems, each with a difficulty rating.

    Adds `rating` and `rating_quantile` (0 = easiest) to the shared record shape. `steps`
    are the <<...>> calculations in the written GSM8K solution, and `answer` is the number
    after its "####". The problem order is not the same as GSM8k-Aug's test split, so
    don't match the two up by position.
    """
    ds = load_dataset("furonghuang-lab/Easy2Hard-Bench", "E2H-GSM8K", split="eval")

    records = []
    for i, row in enumerate(ds):
        records.append({
            "id": f"e2h-gsm8k-{i}",
            "question": row["question"],
            "steps": STEP.findall(row["answer"]),
            "answer": extract_answer(row["answer"]),
            "rating": row["rating"],
            "rating_quantile": row["rating_quantile"],
        })

    return records
