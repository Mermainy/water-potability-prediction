"""Build and start separate serving containers, then verify a real prediction."""
import json
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
from textwrap import dedent

from schema import ROOT


def deploy(root: Path = ROOT) -> dict:
    # Send only serving sources and the model to Docker. This also avoids
    # BuildKit traversing Windows-mounted virtualenvs or restricted test folders.
    files = [
        "code/schema.py", "code/features.py", "code/deployment/docker-compose.yml",
        "code/deployment/api/Dockerfile", "code/deployment/api/requirements.txt",
        "code/deployment/api/main.py", "code/deployment/app/Dockerfile",
        "code/deployment/app/requirements.txt", "code/deployment/app/main.py",
        "models/model.joblib", "models/metrics.json",
    ]
    with TemporaryDirectory(prefix="water-potability-build-") as directory:
        context = Path(directory)
        for relative in files:
            destination = context / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / relative, destination)
        return _deploy(root, context / "code/deployment/docker-compose.yml")


def _deploy(root: Path, compose_file: Path) -> dict:
    metadata = json.loads((root / "models/metrics.json").read_text(encoding="utf-8"))
    compose = ["docker", "compose", "-p", "water-potability-serving",
               "-f", str(compose_file)]
    subprocess.run(compose + ["build"], check=True, timeout=900)
    subprocess.run(compose + ["up", "-d", "--no-build", "--force-recreate", "--wait",
                              "--wait-timeout", "180"], check=True, timeout=240)
    # Execute from inside the API container: localhost here is the serving API,
    # not the scheduler container or a platform-dependent host gateway.
    smoke_script = dedent("""
    import json, urllib.request
    health = json.load(urllib.request.urlopen('http://localhost:8000/health', timeout=10))
    sample = {'ph': 7.0, 'Hardness': 196.0, 'Solids': 22000.0, 'Chloramines': 7.0,
            'Sulfate': 333.0, 'Conductivity': 426.0, 'Organic_carbon': 14.0,
            'Trihalomethanes': 66.0, 'Turbidity': 4.0}
    request = urllib.request.Request('http://localhost:8000/predict',
            data=json.dumps(sample).encode(), headers={'Content-Type': 'application/json'})
    result = json.load(urllib.request.urlopen(request, timeout=10))
    assert result['potability'] in [0, 1]
    assert 0 <= result['probability_potable'] <= 1
    assert health['run_id'] == result['run_id']
    print(json.dumps(result))
    """)
    try:
        result = subprocess.run(compose + ["exec", "-T", "api", "python", "-c", smoke_script],
                                check=True, capture_output=True, text=True, timeout=30)
    except subprocess.CalledProcessError as error:
        details = error.stderr or error.stdout or str(error)
        raise RuntimeError(f"Deployment prediction check failed:\n{details}") from error
    prediction = json.loads(result.stdout)
    if prediction["run_id"] != metadata["run_id"]:
        raise RuntimeError("Serving model does not match the newly trained MLflow run")
    report = {"status": "healthy", "run_id": metadata["run_id"], "smoke_prediction": prediction}
    report_path = root / "models/deployment.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    import mlflow

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", (root / "mlruns").as_uri()))
    with mlflow.start_run(run_id=metadata["run_id"]):
        mlflow.set_tag("deployment_status", "healthy")
        mlflow.log_artifact(str(report_path))
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    deploy()
