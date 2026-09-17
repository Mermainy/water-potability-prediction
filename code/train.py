"""Compare original and engineered features with CV, then package the chosen model."""
import hashlib
import json
import os
from pathlib import Path

import joblib
import mlflow
from mlflow.sklearn import log_model
import pandas as pd
import numpy as np
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate

from schema import FEATURES, ROOT, TARGET
from features import ENGINEERED_FEATURES, WaterFeatures


SCORING = {metric: metric for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]}


def make_model(engineered: bool = False):
    classifier = RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                        class_weight="balanced", random_state=42, n_jobs=2)
    if not engineered:
        return classifier
    # Fit only the residual's expected relationship within each CV training fold.
    # Cleaning remains in preprocessing.py; inference accepts the same nine inputs.
    return Pipeline([
        ("features", WaterFeatures()),
        ("classifier", classifier),
    ])


def compare_features(x_train, y_train) -> dict:
    # Both candidates use exactly the same splits
    # test.csv is excluded from model selection.
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(x_train, y_train))
    comparison = {}
    for name, engineered in [("baseline", False), ("engineered", True)]:
        scores = cross_validate(make_model(engineered), x_train, y_train,
                                cv=folds, scoring=SCORING, n_jobs=1, error_score="raise")
        comparison[name] = {
            metric: {"mean": float(np.mean(scores[f"test_{metric}"])),
                     "std": float(np.std(scores[f"test_{metric}"])),
                     "fold_scores": scores[f"test_{metric}"].tolist()}
            for metric in SCORING
        }
    return comparison


def train(root: Path = ROOT) -> dict:
    processed = root / "data/processed"
    training = pd.read_csv(processed / "train.csv")
    testing = pd.read_csv(processed / "test.csv")
    x_train, y_train = training[FEATURES], training[TARGET]
    x_test, y_test = testing[FEATURES], testing[TARGET]
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", (root / "mlruns").as_uri()))
    mlflow.set_experiment("water-potability")
    output = root / "models"
    output.mkdir(parents=True, exist_ok=True)
    with mlflow.start_run(run_name=os.getenv("AIRFLOW_RUN_ID", "manual")) as run:
        mlflow.log_params({"n_estimators": 200, "min_samples_leaf": 2,
                           "class_weight": "balanced", "random_state": 42,
                           "train_rows": len(training), "test_rows": len(testing),
                           "features": ",".join(FEATURES), "cv_folds": 5,
                           "selection_metric": "cv_mean_roc_auc",
                           "cv_input": "preprocessed_training_data",
                           "engineered_features": ",".join(ENGINEERED_FEATURES)})
        comparison = compare_features(x_train, y_train)
        selected = ("engineered" if comparison["engineered"]["roc_auc"]["mean"] >
                    comparison["baseline"]["roc_auc"]["mean"] else "baseline")
        mlflow.log_param("selected_variant", selected)
        for variant, results in comparison.items():
            mlflow.log_metrics({f"cv_{variant}_{metric}_{stat}": result[stat]
                                for metric, result in results.items() for stat in ["mean", "std"]})
        cv_report = {"folds": 5, "shuffle": True, "random_state": 42,
                     "input": "preprocessed_training_data", "selection_metric": "roc_auc",
                     "engineered_features": ENGINEERED_FEATURES,
                     "selected_variant": selected, "results": comparison}
        cv_path = output / "cv_results.json"
        cv_path.write_text(json.dumps(cv_report, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(cv_path))

        print("Five-fold CV comparison (mean +/- standard deviation):")
        for metric in SCORING:
            baseline, engineered = comparison["baseline"][metric], comparison["engineered"][metric]
            print(f"{metric:10s}: baseline {baseline['mean']:.4f} +/- {baseline['std']:.4f}; "
                  f"engineered {engineered['mean']:.4f} +/- {engineered['std']:.4f}")
        print(f"Selected by mean CV ROC AUC: {selected}")

        model = make_model(engineered=selected == "engineered")
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
        probability = model.predict_proba(x_test)[:, list(model.classes_).index(1)]
        metrics = {
            "accuracy": accuracy_score(y_test, prediction),
            "precision": precision_score(y_test, prediction, zero_division=0),
            "recall": recall_score(y_test, prediction, zero_division=0),
            "f1": f1_score(y_test, prediction, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probability),
        }
        mlflow.log_metrics(metrics)
        log_model(
            model, artifact_path="model", input_example=x_train.head(3),
            serialization_format="cloudpickle",
            signature=infer_signature(x_train, model.predict(x_train)),
            code_paths=[str(ROOT / "code/features.py"), str(ROOT / "code/schema.py")],
            pip_requirements=["scikit-learn==1.6.1", "pandas==2.2.3", "numpy==2.2.4", "joblib==1.4.2"],
        )
        temporary = output / "model.joblib.tmp"
        joblib.dump(model, temporary)
        temporary.replace(output / "model.joblib")
        metadata = {
            "run_id": run.info.run_id, "metrics": metrics, "features": FEATURES,
            "selected_variant": selected,
            "model_features": FEATURES + ENGINEERED_FEATURES if selected == "engineered" else FEATURES,
            "cv_results": comparison,
            "train_sha256": hashlib.sha256((processed / "train.csv").read_bytes()).hexdigest(),
            "model_sha256": hashlib.sha256((output / "model.joblib").read_bytes()).hexdigest(),
        }
        (output / "metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(output / "model.joblib"), artifact_path="package")
        mlflow.log_artifact(str(output / "metrics.json"))
        mlflow.log_artifact(str(processed / "preprocessing.json"))
    print(json.dumps(metadata, indent=2))
    return metadata


if __name__ == "__main__":
    train()
