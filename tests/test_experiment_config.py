import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "arf_nested_cv.json"


def load_config():
    assert CONFIG_PATH.exists()
    return json.loads(CONFIG_PATH.read_text())


def test_outer_cv_uses_fixed_split():
    config = load_config()

    assert config["outer_cv"]["n_splits"] == 5
    assert config["outer_cv"]["seed"] == 42
    assert (
        config["outer_cv"]["split_file"]
        == "data/splits/arf_outer_folds_seed42.csv"
    )


def test_inner_cv_is_separate():
    config = load_config()

    assert config["inner_cv"]["n_splits"] == 3
    assert config["inner_cv"]["shuffle"] is True
    assert config["inner_cv"]["seed"] == 2026


def test_primary_metric_is_auroc():
    config = load_config()
    assert config["primary_metric"] == "auroc"


def test_logistic_grid_is_valid():
    config = load_config()
    grid = config["logistic_regression"]["c_grid"]

    assert len(grid) >= 2
    assert all(value > 0 for value in grid)


def test_lightgbm_grid_is_valid():
    config = load_config()
    grid = config["lightgbm"]["parameter_grid"]

    assert len(grid) == 4

    for parameters in grid:
        assert parameters["num_leaves"] > 1
        assert parameters["min_data_in_leaf"] > 0


def test_feature_set_is_fixed():
    config = load_config()
    assert config["feature_set"] == "12h_sum"
