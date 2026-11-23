"""
Turn a model's text output into a final answer and score it against the gold answer.

Pass `extract_answer` only the text the model generated, not the prompt: the prompt
contains the question's numbers, which the "last number" fallback could pick up.

Two rules:
- `extract_answer` / `is_correct`: lenient, compares numbers (closer to CODI's evaluation).
- `extract_answer_strict` / `is_correct_strict`: COCONUT's exact text match, as in John's
  run.py. Never counts more answers correct than the lenient rule on GSM8K.
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


def _normalize_strict(answer: str) -> str:
    """COCONUT's normalization: no commas, no surrounding spaces, lowercase, no final period."""
    return answer.replace(",", "").strip().lower().rstrip(".")


def extract_answer_strict(text: str) -> str | None:
    """COCONUT's rule (John's run.py): everything after the last "#", normalized.
    Returns None if there is no "#", which COCONUT also always scores as wrong.

    Decode the generated text with `skip_special_tokens=True`, as COCONUT does, so an
    end-of-text token isn't left on the answer."""
    if ANSWER_MARKER not in text:
        return None

    return _normalize_strict(text.split(ANSWER_MARKER)[-1])


def is_correct_strict(pred: str | None, gold: str) -> bool:
    """True if `pred` is exactly the gold answer as text after COCONUT's normalization
    (so "10.0" and "10 dollars" do not match "10")."""
    if pred is None:
        return False

    return pred == _normalize_strict(gold)


def accuracy(preds: list[str | None], golds: list[str], strict: bool = False) -> float:
    """Fraction of predictions that are correct. With `strict=True`, use COCONUT's exact
    text match (`is_correct_strict`) instead of comparing numbers (`is_correct`)."""
    if len(preds) != len(golds):
        raise ValueError(f"{len(preds)} predictions but {len(golds)} gold answers")
    if not golds:
        raise ValueError(f"No examples to score")

    match = is_correct_strict if strict else is_correct
    acc = sum(match(p, g) for p, g in zip(preds, golds)) / len(golds)

    return acc
