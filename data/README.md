# Data

The synthetic generator (`mushroom.make_dataset`) needs nothing downloaded.

For real results, use the public **UCI Secondary Mushroom Dataset**:

- https://archive.ics.uci.edu/dataset/848/secondary+mushroom+dataset
- Semicolon-separated; target column `class` (`e` = edible, `p` = poisonous),
  with numeric measurements (cap-diameter, stem-height, stem-width) and many
  categorical traits.

```python
from mushroom import load_uci, train
df = load_uci("data/MushroomDataset/secondary_data.csv")
pipe, metrics = train(df)
```

The pipeline detects numeric vs categorical columns by dtype, so it adapts to
the full set of real columns without changes.
