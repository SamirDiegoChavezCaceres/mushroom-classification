"""Walkthrough: train on synthetic data, then show sample predictions.

    python scripts/demo.py
"""

from __future__ import annotations

from mushroom import make_dataset, train


def rule(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> None:
    rule("1. Train")
    pipe, metrics = train(make_dataset(n=3000, seed=0))
    print(f"  metrics: {metrics}")

    rule("2. Sample predictions (p = poisonous, e = edible)")
    sample = make_dataset(n=8, seed=99)
    X = sample.drop(columns=["class"])
    actual = sample["class"].tolist()
    pred = pipe.predict(X).tolist()
    for i in range(len(X)):
        row = X.iloc[i]
        mark = "ok" if pred[i] == actual[i] else "MISS"
        print(f"  cap={row.cap_color:<6} gill={row.gill_color:<6} "
              f"bruise={row.does_bruise_or_bleed}  ->  pred={pred[i]} actual={actual[i]}  {mark}")


if __name__ == "__main__":
    main()
