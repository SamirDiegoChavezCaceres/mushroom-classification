# mushroom-classification

Classify mushrooms as edible or poisonous from their traits - a clean,
categorical-heavy tabular ML pipeline.

Runs offline on a synthetic dataset; point it at the public **UCI Secondary
Mushroom Dataset** for real results (see [`data/`](data/)).

## What it shows

- **Dtype-driven preprocessing.** Numeric and categorical columns are detected
  automatically, so the same pipeline trains on the small synthetic frame and on
  the wider real dataset without hand-listing columns.
- **Robust encoding.** Categoricals are one-hot encoded with
  `handle_unknown="ignore"`, so a category never seen during training does not
  crash prediction at serve time.
- **Honest metrics.** Accuracy plus F1 for the poisonous class - the costly
  mistake here is calling a poisonous mushroom edible.

## Run it

```bash
pip install -e .
python scripts/train.py                  # synthetic
python scripts/train.py secondary_data.csv   # real UCI data
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```

Covers column type detection and that the model learns the signal and only ever
predicts valid labels.

## License

MIT.
