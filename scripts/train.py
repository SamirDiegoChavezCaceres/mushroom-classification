"""Train the classifier and print metrics.

    python scripts/train.py                 # synthetic data
    python scripts/train.py path/to/uci.csv # real UCI dataset (semicolon-separated)
"""

from __future__ import annotations

import sys

from mushroom import load_uci, make_dataset, train


def main() -> None:
    if len(sys.argv) > 1:
        df = load_uci(sys.argv[1])
        print(f"loaded {len(df)} rows from {sys.argv[1]}")
    else:
        df = make_dataset()
        print(f"synthetic dataset: {len(df)} rows")
    _, metrics = train(df)
    print("metrics:", metrics)


if __name__ == "__main__":
    main()
