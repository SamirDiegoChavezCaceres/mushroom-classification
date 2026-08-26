from mushroom import make_dataset, train
from mushroom.model import split_columns


def test_column_detection_by_dtype():
    df = make_dataset(n=50)
    numeric, categorical = split_columns(df)
    assert set(numeric) == {"cap_diameter", "stem_height", "stem_width"}
    assert "cap_color" in categorical and "class" not in categorical + numeric


def test_model_learns_and_predicts_labels():
    pipe, metrics = train(make_dataset(n=3000, seed=1))
    assert metrics["accuracy"] > 0.8
    assert metrics["n_categorical"] == 6
    preds = set(pipe.predict(make_dataset(n=20, seed=5).drop(columns=["class"])))
    assert preds <= {"e", "p"}
