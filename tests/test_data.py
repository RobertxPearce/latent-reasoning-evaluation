import json

import pytest
from lre.data.formats import to_codi, to_coconut, write_json
from lre.data.splits import filter_by_steps, train_val_split


def make_example(i: int, n_steps: int) -> dict:
    """A hand-made record in the shape `lre.data.load` returns."""
    return {
        "id": f"ex-{i}",
        "question": f"Question {i}?",
        "steps": [f"<<{j}+1={j + 1}>>" for j in range(n_steps)],
        "answer": str(n_steps),
    }


def test_filter_by_steps():
    examples = [make_example(i, n) for i, n in enumerate([0, 1, 2, 3])]
    kept = filter_by_steps(examples, max_steps=2)
    # zero steps dropped, 1 and 2 kept, 3 dropped
    assert [len(ex["steps"]) for ex in kept] == [1, 2]


def test_filter_by_steps_min_above_max():
    with pytest.raises(ValueError):
        filter_by_steps([make_example(0, 1)], max_steps=1, min_steps=2)


def test_train_val_split():
    examples = [make_example(i, 1) for i in range(10)]
    train, val = train_val_split(examples, n_val=3, seed=0)

    assert len(train) == 7
    assert len(val) == 3
    # no example in both, none lost
    train_ids = {ex["id"] for ex in train}
    val_ids = {ex["id"] for ex in val}
    assert not train_ids & val_ids
    assert train_ids | val_ids == {ex["id"] for ex in examples}


def test_train_val_split_same_seed_same_split():
    examples = [make_example(i, 1) for i in range(10)]
    assert train_val_split(examples, n_val=3, seed=0) == train_val_split(examples, n_val=3, seed=0)


def test_train_val_split_does_not_change_input():
    examples = [make_example(i, 1) for i in range(10)]
    ids_before = [ex["id"] for ex in examples]
    train_val_split(examples, n_val=3, seed=0)
    assert [ex["id"] for ex in examples] == ids_before


def test_to_coconut():
    example = make_example(0, 2)
    # exactly the keys COCONUT's dataset.py reads; id is dropped
    assert to_coconut(example) == {
        "question": example["question"],
        "steps": example["steps"],
        "answer": example["answer"],
    }


def test_to_codi():
    example = make_example(0, 2)
    record = to_codi(example)
    assert set(record) == {"question", "cot", "answer"}
    assert record["cot"] == "<<0+1=1>> <<1+1=2>>"
    # CODI's train.py splits cot on " ", which must give back the steps
    assert record["cot"].split(" ") == example["steps"]


def test_write_json(tmp_path):
    records = [to_coconut(make_example(i, 1)) for i in range(3)]
    # makes the missing "data" folder
    path = tmp_path / "data" / "train.json"
    write_json(records, path)
    with open(path) as f:
        assert json.load(f) == records
