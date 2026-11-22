"""
Pick subsets of the records from `lre.data.load`: the easy training set and a validation set.
"""

import random


def filter_by_steps(examples: list[dict], max_steps: int, min_steps: int = 1) -> list[dict]:
    """Keep the examples with between `min_steps` and `max_steps` reasoning steps (inclusive).

    `min_steps` defaults to 1 because the GSM8k-Aug problems with zero steps are not easy
    ones: they're ordinary problems whose steps are missing from the data.
    """
    if min_steps > max_steps:
        raise ValueError(f"min_steps ({min_steps}) is greater than max_steps ({max_steps})")

    return [ex for ex in examples if min_steps <= len(ex["steps"]) <= max_steps]


def train_val_split(examples: list[dict], n_val: int, seed: int = 0) -> tuple[list[dict], list[dict]]:
    """Randomly split `examples` into (train, val), with `n_val` examples in val.

    The same `seed` always gives the same split, so everyone trains and validates on
    the same examples. `examples` itself is not changed.
    """
    if not 0 < n_val < len(examples):
        raise ValueError(f"n_val must be between 1 and {len(examples) - 1}, got {n_val}")

    shuffled = examples.copy()
    random.Random(seed).shuffle(shuffled)

    val = shuffled[:n_val]
    train = shuffled[n_val:]

    return train, val
