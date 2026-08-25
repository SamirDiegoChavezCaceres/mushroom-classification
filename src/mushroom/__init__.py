"""Edible-vs-poisonous mushroom classification (synthetic data + real UCI loader)."""

from .data import TARGET, load_uci, make_dataset
from .model import build_pipeline, split_columns, train

__all__ = ["make_dataset", "load_uci", "TARGET", "split_columns", "build_pipeline", "train"]
