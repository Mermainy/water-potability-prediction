import json
import runpy
import shutil

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split

from schema import FEATURES, ROOT, TARGET
from train import ENGINEERED_FEATURES, make_model, train
from features import WaterFeatures


def run_preprocessing(root):
    # Execute the original-style script in an isolated repository layout.
    script = root / "code/preprocessing.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / "code/preprocessing.py", script)
    return runpy.run_path(str(script))["metadata"]


@pytest.fixture
def raw_data(tmp_path):
    rng = np.random.default_rng(42)
    centers = np.array([7, 196, 22000, 7, 333, 426, 14, 66, 4])
    scales = np.array([1, 20, 2000, 1, 30, 40, 2, 8, 0.5])
    frame = pd.DataFrame(rng.normal(centers, scales, size=(200, 9)), columns=FEATURES)
    frame[TARGET] = np.tile([0, 1], 100)
    frame.loc[::7, "ph"] = np.nan
    frame.loc[0, "Solids"] = 1e9
    path = tmp_path / "data/raw"
    path.mkdir(parents=True)
    frame.to_csv(path / "water_potability.csv", index=False)
    return tmp_path


def test_cleaning_fits_training_only_and_preserves_holdout(raw_data):
    original = pd.read_csv(raw_data / "data/raw/water_potability.csv")
    original_train, original_test = train_test_split(original, test_size=0.2, random_state=42,
                                                     stratify=original[TARGET])
    metadata = run_preprocessing(raw_data)
    training = pd.read_csv(raw_data / "data/processed/train.csv")
    testing = pd.read_csv(raw_data / "data/processed/test.csv")
    assert list(training.columns) == FEATURES + [TARGET]
    assert not training.isna().any().any()
    assert not testing.isna().any().any()
    assert metadata["imputation_means"]["ph"] == pytest.approx(original_train["ph"].mean())
    assert len(testing) == len(original_test)
    assert metadata["outliers_removed"] > 0
    pd.testing.assert_series_equal(testing[TARGET], original_test[TARGET].reset_index(drop=True))


def test_preprocessing_runs_from_another_working_directory(raw_data, monkeypatch):
    elsewhere = raw_data / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    run_preprocessing(raw_data)
    assert (raw_data / "data/processed/train.csv").is_file()
    assert (raw_data / "data/processed/test.csv").is_file()
    assert not (elsewhere / "data").exists()


def test_training_mlflow_package_and_http_prediction(raw_data, monkeypatch):
    tracking_uri = (raw_data / "mlruns").as_uri()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    run_preprocessing(raw_data)
    metadata = train(raw_data)
    client = MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(metadata["run_id"])
    assert run.info.status == "FINISHED"
    assert {"accuracy", "precision", "recall", "f1", "roc_auc"} <= set(run.data.metrics)
    for variant in ["baseline", "engineered"]:
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            summary = metadata["cv_results"][variant][metric]
            assert len(summary["fold_scores"]) == 5
            assert run.data.metrics[f"cv_{variant}_{metric}_mean"] == pytest.approx(summary["mean"])
    best = max(metadata["cv_results"], key=lambda variant: metadata["cv_results"][variant]["roc_auc"]["mean"])
    assert metadata["cv_results"][metadata["selected_variant"]]["roc_auc"]["mean"] == metadata["cv_results"][best]["roc_auc"]["mean"]
    assert (raw_data / "models/cv_results.json").is_file()
    assert "model" in {a.path for a in client.list_artifacts(metadata["run_id"])}
    model_path = raw_data / "models/model.joblib"
    model = joblib.load(model_path)
    sample = pd.read_csv(raw_data / "data/processed/test.csv")[FEATURES].iloc[0].to_dict()
    expected = int(model.predict(pd.DataFrame([sample], columns=FEATURES))[0])
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    from deployment.api.main import app

    with TestClient(app) as api:
        assert api.get("/health").json()["run_id"] == metadata["run_id"]
        response = api.post("/predict", json=sample)
        assert response.status_code == 200
        result = response.json()
        assert result["potability"] == expected
        assert 0 <= result["probability_potable"] <= 1
        assert result["run_id"] == metadata["run_id"]
        assert api.post("/predict", json={}).status_code == 422
        assert api.post("/predict", json={**sample, "ph": 20}).status_code == 422
        assert api.post("/predict", json={**sample, "Solids": -1}).status_code == 422
        assert api.post("/predict", json={**sample, "unknown": 3}).status_code == 422
    assert json.loads((raw_data / "models/metrics.json").read_text())["run_id"] == metadata["run_id"]


def test_engineered_model_roundtrip_and_api(raw_data, monkeypatch):
    # Exercise engineered serving even when CV happens to choose the baseline.
    run_preprocessing(raw_data)
    training = pd.read_csv(raw_data / "data/processed/train.csv")
    model = make_model(engineered=True).fit(training[FEATURES], training[TARGET])
    transformed = model[:-1].transform(training[FEATURES].head(1))
    assert list(transformed.columns) == FEATURES + ENGINEERED_FEATURES
    assert transformed["chloramines_per_organic_carbon"].iloc[0] == pytest.approx(
        training["Chloramines"].iloc[0] / training["Organic_carbon"].iloc[0])
    assert transformed["trihalomethanes_per_organic_carbon"].iloc[0] == pytest.approx(
        training["Trihalomethanes"].iloc[0] / 1000 / training["Organic_carbon"].iloc[0])
    output = raw_data / "models"
    output.mkdir()
    path = output / "model.joblib"
    joblib.dump(model, path)
    (output / "metrics.json").write_text(json.dumps({"run_id": "engineered-smoke"}))
    monkeypatch.setenv("MODEL_PATH", str(path))
    from deployment.api.main import app

    sample = training[FEATURES].iloc[0].to_dict()
    with TestClient(app) as api:
        result = api.post("/predict", json=sample)
        assert result.status_code == 200
        assert result.json()["potability"] == int(model.predict(training[FEATURES].head(1))[0])
        assert result.json()["run_id"] == "engineered-smoke"
        zero_denominators = {**sample, "Conductivity": 0, "Organic_carbon": 0}
        assert api.post("/predict", json=zero_denominators).status_code == 200


def test_residual_relationship_uses_training_rows_only():
    training = pd.DataFrame(1.0, index=range(4), columns=FEATURES)
    training["Conductivity"] = [1, 2, 3, 4]
    training["Solids"] = [5, 7, 9, 11]  # Solids = 2 * conductivity + 3.
    transformer = WaterFeatures().fit(training)
    validation = training.iloc[:1].copy()
    validation["Conductivity"] = 5
    validation["Solids"] = 100
    result = transformer.transform(validation)
    assert result["solids_conductivity_residual"].iloc[0] == pytest.approx(87)
    assert transformer.solids_model_.coef_[0] == pytest.approx(2)
    assert transformer.solids_model_.intercept_ == pytest.approx(3)
    validation[["Conductivity", "Organic_carbon"]] = 0
    assert np.isfinite(transformer.transform(validation).to_numpy()).all()
