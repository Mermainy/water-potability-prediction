# How to run the assignment

## 1. Prerequisites

Use Docker Desktop with **Linux containers**, a working Docker Compose v2 plugin (2.20 or newer), and internet access for the initial image/dependency downloads. On Windows enable the WSL 2 backend. Allocate about 8 GB of memory to Docker if possible.

Open PowerShell in the repository root:

```powershell
Set-Location 'C:\Users\КАРИНА\PythonProjects\water-potability-prediction'
docker info
docker compose version
```

`docker info` must display a Server section. If it reports `docker_engine` does not exist, start Docker Desktop and wait for the engine to be ready. You do not need to install Airflow into your Windows Python environment.

The images use Python 3.11, Airflow 3.3.1, and MLflow 2.22.0. This is a local assignment setup with demonstration credentials. The scheduler mounts the Docker socket to build images and manage the two serving containers on your Docker engine; use this stack on a trusted development machine.

## 2. Start the infrastructure

When upgrading an existing Airflow 2 stack, back up its PostgreSQL metadata database first, then run `docker compose down --remove-orphans` to stop the old services while preserving volumes. Follow the startup commands below to build Airflow 3 and migrate the database. The separate `airflow-dag-processor` service parses the DAG; the API server provides the UI and task execution API.

Run these commands in order from the repository root:

```powershell
Copy-Item .env.example .env
docker compose config --quiet
docker compose build airflow-init mlflow
docker compose up airflow-init
docker compose up -d airflow-api-server airflow-dag-processor airflow-scheduler
docker compose ps
```

If you already have `.env`, preserve it and edit it instead of copying over it. `airflow-init` must finish with exit code 0. It migrates the Airflow database, creates the `airflow` user, and initializes writable artifact volumes. PostgreSQL and MLflow start as dependencies. Initial builds may take several minutes.

Your existing `data/raw/water_potability.csv` is copied into the data volume during initialization if the volume has no dataset yet. On a fresh checkout without that CSV, the DAG downloads the public Kaggle dataset automatically and reuses it on subsequent runs. If Kaggle is unavailable or requests authentication, download `water_potability.csv` manually from the dataset page, put it in `data/raw/`, and rerun `docker compose up airflow-init` before starting the scheduler (or use the dataset replacement command below).

Open these pages:

| Service | URL | Login |
| --- | --- | --- |
| Airflow | http://localhost:8080 | `airflow` / `airflow` |
| MLflow | http://localhost:5000 | None |
| FastAPI Swagger | http://localhost:8000/docs | Available after deployment |
| Web application | http://localhost:8501 | Available after deployment |

## 3. Run the complete pipeline

The DAG is unpaused by default and runs automatically at five-minute UTC boundaries (`*/5 * * * *`). It has four sequential tasks:

1. `load_data`: read/reuse/download the raw CSV.
2. `prepare_data`: clean the data and save `train.csv`, `test.csv`, and `preprocessing.json`.
3. `train_and_evaluate`: compare random forests with original and engineered features using five-fold stratified CV, select by mean CV ROC AUC, fit the selected version, log metrics/model to MLflow, and save `model.joblib`, `metrics.json`, and `cv_results.json`.
4. `build_and_deploy`: build both serving images, start/recreate their separate containers, wait for health, and verify a real prediction uses the newly trained run.

To start a run immediately instead of waiting for the next schedule boundary:

```powershell
docker compose exec airflow-scheduler airflow dags list-import-errors
docker compose exec airflow-scheduler airflow dags trigger water_potability_pipeline
```

You can also select `water_potability_pipeline` in Airflow and click **Trigger DAG**. Wait until all four tasks are green. The first deployment is slower because it installs API and application dependencies; subsequent builds reuse Docker layers. One active run at a time prevents overlapping writes and deployments. Scheduled runs still evaluate and deploy even when the source dataset has not changed.

Follow task logs in Airflow. To inspect infrastructure logs:

```powershell
docker compose logs --tail 100 airflow-scheduler airflow-dag-processor airflow-api-server mlflow
```

## 4. Check the containers and make predictions

The serving containers belong to a separate Compose project. Inspect them with:

```powershell
docker compose -f code/deployment/docker-compose.yml ps
docker compose -f code/deployment/docker-compose.yml logs --tail 100 api app
Invoke-RestMethod http://localhost:8000/health
```

Visit http://localhost:8501, fill the nine measurement fields (the default sample is ready to use), and click **Predict**. The app displays the predicted class, estimated probability, and MLflow run ID. It sends the request to the API container at `http://api:8000`.

For a direct API prediction in PowerShell:

```powershell
$sample = @{
    ph = 7.0
    Hardness = 196.0
    Solids = 22000.0
    Chloramines = 7.0
    Sulfate = 333.0
    Conductivity = 426.0
    Organic_carbon = 14.0
    Trihalomethanes = 66.0
    Turbidity = 4.0
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict -ContentType 'application/json' -Body $sample
```

The response has `potability` (`0` or `1`), `label`, `probability_potable`, and `run_id`. Inputs are required; unknown fields, nonfinite numbers, negative measurements, or a pH outside 0–14 return HTTP 422. This is an educational classifier, not a laboratory assessment.

In MLflow, open experiment **water-potability**. Each training run contains the five testing metrics, training parameters, model signature, packaged model, cleaning statistics, and file hashes. The `cv_baseline_*` and `cv_engineered_*` metrics contain the five-fold means and standard deviations for both feature sets; `cv_results.json` includes individual fold scores. The `selected_variant` parameter identifies the deployed candidate, chosen by mean CV ROC AUC (ties keep the baseline). Successfully deployed runs additionally have tag `deployment_status=healthy` and artifact `deployment.json`. Match the run ID with `/health` or the web app to demonstrate that the new model was deployed.

Feature engineering adds three ratios (solids / conductivity, chloramines / organic carbon, and trihalomethanes / organic carbon) and a solids–conductivity residual. The residual is observed solids minus the prediction from a linear regression fitted within each CV training fold. THMs are converted from micrograms/L to mg/L before division by organic carbon (ppm treated as mg/L for dilute water); ratio denominators have a numerical floor of `1e-6` to handle zero inputs. These are exploratory relationships, not calibrated chemical laws. The custom transformer is saved with the model, its source is archived in MLflow, and `features.py` is included in the API image.

Both candidates use the same folds and forest settings. CV uses only the processed training CSV; the test CSV is used once for the selected candidate's final evaluation. As requested, imputation and outlier removal are performed only in preprocessing. Consequently, the CV results describe the feature comparison on already cleaned data, with cleaning statistics fitted before the folds were constructed. They are not a validation estimate of the complete preprocessing procedure.

## 5. Inspect or export the generated files

In the automatic Docker workflow, files are written into named volumes mounted inside the scheduler. They are not written into the host's `data/processed` or `models` directories.

```powershell
docker compose exec airflow-scheduler ls -lh /opt/project/data/processed /opt/project/models
docker compose exec airflow-scheduler cat /opt/project/models/metrics.json
docker compose exec airflow-scheduler cat /opt/project/models/cv_results.json
docker compose exec airflow-scheduler cat /opt/project/models/deployment.json
```

Export a snapshot to the repository if desired:

```powershell
New-Item -ItemType Directory -Force data/processed, models
docker compose cp airflow-scheduler:/opt/project/data/processed/. ./data/processed/
docker compose cp airflow-scheduler:/opt/project/models/. ./models/
```

The raw data, train/test files, model packages, and runtime state are ignored by Git. The API image contains the trained model and its metadata; it does not require host bind mounts to serve predictions. Every deployment briefly recreates the serving containers.

To replace the raw dataset after initialization, pause the DAG in Airflow and wait for any active run to finish, then:

```powershell
docker compose cp ./data/raw/water_potability.csv airflow-scheduler:/opt/project/data/raw/water_potability.csv
docker compose exec --user 0 airflow-scheduler chown 50000:0 /opt/project/data/raw/water_potability.csv
docker compose exec airflow-scheduler airflow dags unpause water_potability_pipeline
docker compose exec airflow-scheduler airflow dags trigger water_potability_pipeline
```

## 6. Change the interval if a run is too slow

The assignment allows a longer interval when a complete run takes longer than five minutes. In `.env`, change:

```dotenv
PIPELINE_SCHEDULE=*/10 * * * *
```

Then recreate the Airflow services:

```powershell
docker compose up -d --force-recreate airflow-scheduler airflow-dag-processor airflow-api-server
```

Leave the default five-minute schedule if steady-state runs complete within it. Wait for any active run to finish before recreating the scheduler.

## 7. Optional local Python execution and tests

The Docker workflow above is the complete automated assignment. For debugging the Python stages locally, create a separate environment so the notebook environment is preserved. Python 3.11 matches the images; your installed Python 3.13 was also verified for these local tests:

```powershell
python -m venv .venv-assignment
.\.venv-assignment\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv-assignment\Scripts\python.exe -m pytest -q
```

Tests cover fitting cleaning statistics only on training rows, retaining the holdout, removing training outliers, running preprocessing from another working directory, logging CV results and an MLflow model, reloading the package, making HTTP predictions, and validating API inputs. An additional test explicitly serves the engineered model to verify its feature transformations survive packaging even when CV selects the baseline, including zero-denominator inputs. A fifth test checks that the residual estimator retains its training-only relationship when transforming validation data. Tests use temporary datasets and tracking storage.

After the Docker deployment is healthy, optionally verify the actual Streamlit form submits to the running API and displays its response:

```powershell
.\.venv-assignment\Scripts\python.exe tests/check_live_app.py
```

This uses Streamlit's application testing interface to fill the default form, click Predict, and check the displayed label and serving MLflow run ID. It requires the API to be running on host port 8000. Set `SMOKE_API_URL` if you changed that port.

Run the stages locally against your raw dataset:

```powershell
$env:MLFLOW_TRACKING_URI = 'http://localhost:5000'
.\.venv-assignment\Scripts\python.exe code/download.py
.\.venv-assignment\Scripts\python.exe code/preprocessing.py
.\.venv-assignment\Scripts\python.exe code/train.py
.\.venv-assignment\Scripts\python.exe code/deploy.py
```

These commands save files in the host's `data/processed` and `models`. `code/deploy.py` still requires Docker; it performs the same build, startup, and prediction verification as the DAG. If MLflow is not running, omit/unset `MLFLOW_TRACKING_URI` to use local `mlruns` storage. Do not manually deploy while an automatic DAG deployment is active.

## 8. Stop and restart

Stop the scheduler first to prevent a scheduled run from redeploying the app:

```powershell
docker compose stop airflow-scheduler
docker compose -f code/deployment/docker-compose.yml down
docker compose down
```

These commands preserve data volumes. Restart the pipeline with:

```powershell
docker compose up -d airflow-api-server airflow-dag-processor airflow-scheduler
```

The next successful DAG run restores the serving containers. For a complete reset, first stop the serving project and then run `docker compose down --volumes`; this deletes the pipeline's dataset, models, MLflow history, Airflow history, and logs. Reinitialize with the commands in section 2.

## 9. Troubleshooting

| Symptom | Action |
| --- | --- |
| Docker engine connection fails | Start Docker Desktop, enable Linux containers, and confirm `docker info` works. |
| Build cannot reach package/image servers | Check Docker internet access/proxy settings and retry the build. |
| API/app URLs fail before the first run | Wait for `build_and_deploy` to finish; they are created by the DAG. |
| DAG is missing or not running | Inspect `airflow dags list-import-errors`, scheduler logs, and the DAG pause state. |
| Training cannot reach MLflow | Check `docker compose ps` and MLflow logs; the in-container URI must be `http://mlflow:5000`. |
| Deployment has Docker socket permission denied on Linux | Set `DOCKER_GID` in `.env` to the group ID from `stat -c '%g' /var/run/docker.sock`, then recreate the scheduler. Windows Docker Desktop normally uses `0`. |
| Airflow repeatedly restarts | Check memory allocation and PostgreSQL health; use around 8 GB for this stack. |
| Ports 5000, 8080, 8000, or 8501 are occupied | Stop the process using the port or adjust the published host port in the corresponding Compose file and browser URL. |
| Serving run ID is older than the latest training run | Inspect the deployment task: failed training/deployment prevents a run from being marked successfully deployed. |

## 10. Demonstration and submission

1. Show the Airflow DAG graph and a successful run with all four tasks green.
2. Show a second scheduled run about five minutes later (or your documented longer interval).
3. Show the train/test outputs and packaged model using the inspection commands.
4. Show the testing metrics and model artifacts in MLflow.
5. Show `docker compose -f code/deployment/docker-compose.yml ps` with separate healthy API and app containers.
6. Predict through Streamlit and match its MLflow run ID to the deployed training run.
7. Commit the source, configurations, requirements, and documentation to your public GitHub repository and submit its link. No generated model file is needed in GitHub: the DAG produces it.

Do not commit `.env`, credentials, virtual environments, or generated runtime files. Publishing the repository is a separate action you perform with your GitHub account.

## Validation performed

Feature-comparison update on 2026-09-17: all five tests pass, including engineered-model packaging, HTTP inference, zero denominators, and training-only residual fitting. The training stage saves the CV comparison and selects its final model using training CV results. On the supplied training data, mean CV ROC AUC is 0.6856 for the baseline and 0.6664 with the four relationship features, so the baseline is selected. Its final test accuracy is 0.6494 and ROC AUC is 0.6617. The full CV comparison is in README and generated `models/cv_results.json`. The earlier runtime timings below describe the original implementation before this update.

The following runtime checks were performed on Airflow 2.10.5 on 2026-09-16 with Windows Docker Desktop and Linux containers; they have not been rerun after the Airflow 3 upgrade:

- Both Compose configurations validate, and the infrastructure and serving images build.
- Airflow initializes successfully, imports the DAG without errors, and completes two consecutive scheduled runs through all four tasks. The next run starts automatically at the five-minute boundary; the cached complete run takes about 40 seconds.
- MLflow records the model and holdout metrics; API and app are separate healthy containers.
- The live Streamlit form smoke check submits a prediction to the Docker API and displays the matching MLflow run ID.
- The isolated local Python 3.13 environment passes all three integration tests and `pip check`. Runtime images use Python 3.11.
- The supplied dataset yields 2,595 training rows, 656 holdout rows, and 25 removed training outliers. Holdout accuracy is approximately 0.649 and ROC AUC approximately 0.662.

References: [Airflow 3.3.1 Docker setup](https://airflow.apache.org/docs/apache-airflow/3.3.1/howto/docker-compose/index.html), [MLflow tracking server](https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/), [Docker Compose build configuration](https://docs.docker.com/reference/compose-file/build/).
