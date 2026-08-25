"""Mushroom data: a synthetic generator for offline runs, plus a loader for the
real public UCI dataset.

The synthetic generator mimics the shape of the UCI Secondary Mushroom Dataset
(a mix of a few numeric measurements and several categorical traits, target
``class`` = edible ``e`` / poisonous ``p``) so the demo and tests need nothing
downloaded. For real results, point ``load_uci`` at the public CSV (see
``data/README.md``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "class"

_CAP_SHAPES = ["bell", "conical", "convex", "flat", "sunken"]
_COLORS = ["brown", "yellow", "white", "red", "gray"]
_HABITATS = ["grasses", "woods", "leaves", "paths"]
_SEASONS = ["spring", "summer", "autumn", "winter"]


def make_dataset(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cap_diameter = rng.normal(6, 3, n).clip(1, 20)
    stem_height = rng.normal(7, 3, n).clip(1, 25)
    stem_width = rng.normal(12, 6, n).clip(1, 40)
    cap_shape = rng.choice(_CAP_SHAPES, n)
    cap_color = rng.choice(_COLORS, n)
    gill_color = rng.choice(_COLORS, n)
    habitat = rng.choice(_HABITATS, n)
    season = rng.choice(_SEASONS, n)
    bruises = rng.choice(["t", "f"], n)

    # A learnable rule for poisonousness. Coefficients are strong enough that
    # most rows sit near 0 or 1 (separable), with some genuine overlap.
    logit = 1.7 * (
        -2.5
        + 3.0 * (cap_color == "red")
        + 2.5 * (gill_color == "white")
        + 2.0 * (bruises == "t")
        + 1.5 * np.isin(cap_shape, ["conical", "sunken"])
        + 1.5 * (cap_diameter > 9)
        - 2.0 * (habitat == "grasses")
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    poisonous = rng.random(n) < prob

    return pd.DataFrame(
        {
            TARGET: np.where(poisonous, "p", "e"),
            "cap_diameter": cap_diameter.round(2),
            "stem_height": stem_height.round(2),
            "stem_width": stem_width.round(2),
            "cap_shape": cap_shape,
            "cap_color": cap_color,
            "gill_color": gill_color,
            "habitat": habitat,
            "season": season,
            "does_bruise_or_bleed": bruises,
        }
    )


def load_uci(path: str) -> pd.DataFrame:
    """Load the real UCI Secondary Mushroom Dataset (semicolon-separated)."""
    return pd.read_csv(path, sep=";")
