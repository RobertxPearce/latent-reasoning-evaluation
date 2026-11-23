import pytest
from lre.metrics import accuracy, extract_answer, extract_answer_strict, is_correct, is_correct_strict

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


@pytest.mark.parametrize(
    "text, expected",
    [
        ("### 10", "10"),
        # commas removed, lowercased, final period dropped
        ("### 1,000", "1000"),
        ("### True.", "true"),
        # everything after the last "#" is kept, not just the number
        ("### 10 dollars", "10 dollars"),
        ("<<5*2=10>>\n### 10", "10"),
        # no marker: COCONUT scores it wrong
        ("= 10", None),
    ],
)
def test_extract_answer_strict(text, expected):
    assert extract_answer_strict(text) == expected


@pytest.mark.parametrize(
    "text, gold, strict, lenient",
    [
        # the cases where the two rules disagree, and one where they agree
        ("### 10", "10", True, True),
        ("### 10.0", "10", False, True),
        ("### $10", "10", False, True),
        ("### 10 dollars", "10", False, True),
        ("= 10", "10", False, True),
        # gold with a comma still matches
        ("### 1000", "1,000", True, True),
    ],
)
def test_strict_vs_lenient(text, gold, strict, lenient):
    assert is_correct_strict(extract_answer_strict(text), gold) == strict
    assert is_correct(extract_answer(text), gold) == lenient


def test_accuracy_strict():
    texts = ["### 10", "### 10.0", "= 10"]
    golds = ["10", "10", "10"]
    assert accuracy([extract_answer_strict(t) for t in texts], golds, strict=True) == pytest.approx(1 / 3)
    assert accuracy([extract_answer(t) for t in texts], golds) == pytest.approx(1)
