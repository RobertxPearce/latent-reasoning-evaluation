import pytest
from lre.metrics import accuracy, extract_answer, is_correct

@pytest.mark.parametrize(
    "text, expected",
    [
        # GSM8K gold format
        ("She earned 0.2 x 50 = 10.\n#### 10", "10"),
        # GSM8k-Aug gold: bare number
        ("750", "750"),
        # commas removed
        ("### 1,000", "1000"),
        # "$" removed
        ("### $10", "10"),
        # negative
        ("### -3", "-3"),
        # decimal
        ("### 0.5", "0.5"),
        # after a marker: first number wins
        ("### 12 (checked 3 times)", "12"),
        # no marker: last number, no trailing "."
        ("The answer is 42.", "42"),
        # no number at all
        ("I don't know", None),
    ],
)
def test_extract_answer(text, expected):
    assert extract_answer(text) == expected


@pytest.mark.parametrize(
    "pred, gold, expected",
    [
        ("10", "10", True),
        # same number, different format
        ("10.0", "10", True),
        ("$10", "10", True),
        ("1000", "1,000", True),
        ("11", "10", False),
        # nothing extracted counts as wrong
        (None, "10", False),
    ],
)
def test_is_correct(pred, gold, expected):
    assert is_correct(pred, gold) == expected


def test_accuracy():
    preds = ["10", "11", None]
    golds = ["10", "10", "10"]
    assert accuracy(preds, golds) == pytest.approx(1 / 3)