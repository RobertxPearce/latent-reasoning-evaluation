"""
Turn a model's text output into a final answer and score it against the gold answer.

Pass `extract_answer` only the text the model generated, not the prompt: the prompt
contains the question's numbers, which the "last number" fallback could pick up.
"""

import re

# Expressions to identify answer
NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
# GSM8K gold solutions end with "#### <answer>"
ANSWER_MARKER = "#"


def extract_answer(text: str) -> str | None:
    """Return the final numeric answer in `text` as a string (e.g. "10", "-3", "0.5"),
    with commas and "$" removed, or None if there is no number."""
    text = text.replace(",","")
    
    if ANSWER_MARKER in text:
        after_marker = text.split(ANSWER_MARKER)[-1]
        match = NUMBER.search(after_marker)
        if match:
            return match.group()
        else:
            return None
    
    numbers = NUMBER.findall(text)
    if numbers:
        return numbers[-1]
    else:
        return None


def _to_number(answer: str) -> float | None:
    """Parse an answer string like "1,000", "$10" or "10.0" as a number; None if it isn't one."""
    cleaned = answer.replace(",", "").replace("$", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def is_correct(pred: str | None, gold: str) -> bool:
    """True if `pred` and `gold` are the same number (so "10.0" matches "10")."""
    if pred is None:
        return False
    
    pred_number = _to_number(pred)
    gold_number = _to_number(gold)
    if pred_number is None or gold_number is None:
        return False
    
    return pred_number == gold_number


def accuracy(preds: list[str | None], golds: list[str]) -> float:
    """Fraction of predictions that are correct."""
    if len(preds) != len(golds):
        raise ValueError(f"{len(preds)} predictions but {len(golds)} gold answers")
    if not golds:
        raise ValueError(f"No examples to score")
    
    acc = sum(is_correct(p, g) for p, g in zip(preds, golds)) / len(golds)
    
    return acc
