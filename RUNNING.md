# Run the project for the first time

This guide takes you from a downloaded copy of the project to a prediction in your browser. The commands use **Windows PowerShell**. Run them from the project folder, one at a time, and check each result before continuing.

Docker runs Python, Airflow, PostgreSQL, MLflow, the prediction API, and the web app. You do not need to install Python or Airflow locally for this workflow.

## 1. Install and start Docker

Install Docker Desktop using the [official Windows installation guide](https://docs.docker.com/desktop/setup/install/windows-install/). Use the WSL 2 backend and **Linux containers**. Follow the installer's instructions if WSL needs installation or a restart.

Open Docker Desktop and wait until its engine is running. Keep Docker Desktop open while using the project. Allow several GB of free disk space for images and dependencies, and about 8 GB of memory for the stack where possible. Internet access is needed for the initial downloads.

The containers use Python 3.11, Airflow 3.3.1, MLflow 2.22.0, and PostgreSQL 16. The credentials and settings are intended for a local demonstration. The Airflow scheduler accesses Docker to build and start the API and app.

## 2. Open the project folder

Download and extract the repository ZIP, or clone the repository. Open the extracted or cloned folder in VS Code, then choose **Terminal > New Terminal** and select PowerShell.

The folder must contain `docker-compose.yaml`, `.env.example`, `code`, and `services`. This is the **repository root**. All commands below assume your terminal is there.

Check the location and Docker:

```powershell
Test-Path .\docker-compose.yaml
docker info
docker compose version
```

Expected results:

- `Test-Path` prints `True`. If it prints `False`, open the folder containing `docker-compose.yaml` and start a terminal there.
- `docker info` displays Client and Server information. An engine connection error means Docker Desktop is not ready.
- `docker compose version` displays a Compose v2 version. Use a current Docker Desktop installation with `docker compose wait` available.

## 3. Create the configuration file

Copy the example configuration only if `.env` does not exist:

```powershell
if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
}
```

For Windows with Docker Desktop, leave the defaults:

```dotenv
DOCKER_GID=0
PIPELINE_SCHEDULE=*/5 * * * *
```

The schedule means the pipeline runs every five minutes. You can also start a run manually. Keep an existing `.env` when restarting the project.

Validate the configuration:

```powershell
docker compose config --quiet
```

Success normally produces no output. Resolve any reported error before building.

## 4. Build and initialize the infrastructure

Build the Airflow and MLflow images:

```powershell
docker compose build airflow-init mlflow
```

The first build can take several minutes while Docker downloads images and installs dependencies. Wait for the command to finish successfully.

Start initialization in the background, wait for it to finish, and inspect its exit status:

```powershell
docker compose up -d airflow-init
docker compose wait airflow-init
docker compose ps -a airflow-init
```

The `airflow-init` container must show **Exited (0)**. This is normal: initialization is a one-time task, not a service that stays running. It prepares storage, migrates the Airflow database, and creates the Airflow login. PostgreSQL and MLflow start automatically as dependencies.

If initialization exits with a different code, inspect its logs before continuing:

```powershell
docker compose logs --tail 100 airflow-init
```

You do not have to download the dataset in advance. The pipeline downloads `water_potability.csv` automatically on its first run. If a copy already exists at `data/raw/water_potability.csv`, initialization copies it into Docker storage when that storage has no dataset yet.

## 5. Start Airflow and open its page

```powershell
docker compose up -d airflow-api-server airflow-dag-processor airflow-scheduler
docker compose ps
```

The `-d` option keeps the containers running in the background, so closing the terminal does not stop them. Allow time for startup; run `docker compose ps` again if a service is still starting.

These services should eventually be running and healthy:

| Service | What it does |
| --- | --- |
| `postgres` | Stores Airflow's scheduling history, task states, and users. |
| `mlflow` | Stores training metrics and model artifacts. |
| `airflow-api-server` | Provides the Airflow browser interface and execution API. |
| `airflow-dag-processor` | Reads the pipeline definition and registers it with Airflow. |
| `airflow-scheduler` | Schedules runs and executes their tasks. |

Open [Airflow](http://localhost:8080) and sign in with username **airflow** and password **airflow**. Find **water_potability_pipeline**. Airflow calls a pipeline a **DAG**.

You can also open [MLflow](http://localhost:5000). Its `water-potability` experiment appears after training begins. No MLflow login is required.

The prediction API and web app are created by the pipeline's last task. Their pages become available after the first successful deployment.

## 6. Complete the first pipeline run

The DAG is enabled by default. It starts automatically at five-minute boundaries, such as 12:00, 12:05, and 12:10. Allow time for the DAG processor to discover it.

If a run is already active, watch that run. Otherwise, open the DAG and click **Trigger DAG** to start immediately. Make sure the DAG is unpaused if you want automatic runs.

Alternatively, check for import errors and trigger a run from PowerShell:

```powershell
docker compose exec airflow-scheduler airflow dags list-import-errors
docker compose exec airflow-scheduler airflow dags trigger water_potability_pipeline
```

The import-error check should report no errors. If the DAG is not found yet, wait briefly and check the DAG processor logs in section 10.

Open the run's task graph or grid to follow these tasks in order:

| Task | Expected result |
| --- | --- |
| `load_data` | Downloads or reuses the raw dataset. |
| `prepare_data` | Saves cleaned training data, test data, and preprocessing metadata. |
| `train_and_evaluate` | Compares two feature sets, trains the selected model, and records metrics in MLflow. |
| `build_and_deploy` | Builds and starts the API and app, then verifies a prediction from the new model. |

Wait until **all four tasks and the run show success**. The first deployment takes longer because it builds two more images. Later builds reuse Docker's cache. Only one run executes at a time; triggering another while one is active can leave the new run queued.

If a task fails, open it and read its **Logs**. Later tasks cannot complete until the earlier task succeeds. Airflow retries a failed task once after 30 seconds, so it may temporarily show a retry state.

## 7. Make your first prediction

After `build_and_deploy` succeeds, check the serving containers:

```powershell
docker compose -f code/deployment/docker-compose.yml ps
Invoke-RestMethod http://localhost:8000/health
```

Both `api` and `app` should be running and healthy. The health request should return `status` equal to `ok` and a `run_id` identifying the deployed training run.

Open [the web app](http://localhost:8501). Leave the nine measurement fields at their default values and click **Predict**. The page should display a label, an estimated probability of potability, and the MLflow run ID.

You can explore the API at [its interactive documentation](http://localhost:8000/docs). Expand **POST /predict**, choose **Try it out**, enter all nine measurements, and click **Execute**. The fields are `ph`, `Hardness`, `Solids`, `Chloramines`, `Sulfate`, `Conductivity`, `Organic_carbon`, `Trihalomethanes`, and `Turbidity`.

All measurements must be finite and nonnegative; pH must be between 0 and 14. Missing or unknown fields and invalid values produce an HTTP 422 validation response.

This is an educational model. Its prediction does not replace laboratory testing of drinking water.

## 8. Check the training results

In [MLflow](http://localhost:5000), open the **water-potability** experiment and select the run whose ID matches the web app or API health response.

The run contains accuracy, precision, recall, F1, and ROC AUC metrics, along with the model and preprocessing artifacts. The `selected_variant` parameter shows whether the original or engineered features were selected. The `cv_results.json` artifact contains the five-fold comparison. A successful deployment adds the tag `deployment_status=healthy` and artifact `deployment.json`.

For the model comparison and feature-engineering details, see [README.md](README.md).

Your first launch is complete when the four Airflow tasks succeed, both serving containers are healthy, and the web app displays a prediction from the matching MLflow run.

## 9. Stop and restart the project

Before stopping, pause the DAG in Airflow and let any active run finish. Then run:

```powershell
docker compose stop airflow-scheduler
docker compose -f code/deployment/docker-compose.yml down
docker compose down
```

There are two Compose projects: the root file manages Airflow, PostgreSQL, and MLflow; the file under `code/deployment` manages the API and app. Stopping both shuts down the whole project. These commands preserve the named volumes containing data, models, and history.

For the next launch, start Docker Desktop, open a terminal in the repository root, and run:

```powershell
docker compose up -d airflow-api-server airflow-dag-processor airflow-scheduler
```

If a previous deployment completed successfully, restore its API and app using the already built images:

```powershell
docker compose -f code/deployment/docker-compose.yml up -d --no-build
```

Unpause the DAG in Airflow to resume automatic runs. Rebuilding images and copying `.env` are not needed for an ordinary restart. If the first deployment never succeeded, trigger the pipeline instead of trying to restore serving images.

## 10. Troubleshooting

For startup or pipeline errors:

```powershell
docker compose ps -a
docker compose logs --tail 100 airflow-init airflow-api-server airflow-dag-processor airflow-scheduler mlflow postgres
```

For API or web app errors after deployment:

```powershell
docker compose -f code/deployment/docker-compose.yml logs --tail 100 api app
```

| Problem | What to check |
| --- | --- |
| `docker` is not recognized | Install Docker Desktop, then reopen the terminal so it sees the updated PATH. |
| Docker engine connection error | Start Docker Desktop, wait for its engine, and rerun `docker info`. Use Linux containers. |
| No Compose configuration file found | Open a terminal in the repository root. `Test-Path .\docker-compose.yaml` must return `True`. |
| `docker compose wait` is not recognized | Update Docker Desktop to obtain a current Compose v2 plugin. |
| Image or dependency download fails | Check internet/proxy settings and retry the build command. |
| Initialization exits with a nonzero code | Read `airflow-init` logs. After resolving the error, rerun `docker compose up -d --force-recreate airflow-init`, then wait and check its exit status again. |
| Airflow page does not open | Check the API server's health and logs. PostgreSQL and initialization must succeed first. |
| DAG is missing | Wait for discovery, run `airflow dags list-import-errors` as shown above, and inspect DAG processor logs. |
| DAG has no new runs | Check that it is unpaused. An active run can keep another run queued. |
| API or app page does not open | Check whether `build_and_deploy` succeeded, then inspect serving container health and logs. |
| Training cannot reach MLflow | Check its container and logs. Containers use `http://mlflow:5000`; your browser uses `http://localhost:5000`. |
| Airflow restarts repeatedly | Check memory availability, PostgreSQL health, and the failing service's logs. |
| A port is already occupied | Free host port 8080 for Airflow, 5000 for MLflow, 8000 for the API, or 8501 for the app. |
| Deployed run ID is older than the newest training run | Check the deployment task. A new model is served only after successful deployment. |

### If the dataset download fails

Download and extract `water_potability.csv` from [the Kaggle dataset](https://www.kaggle.com/datasets/adityakadiwal/water-potability). Create its local folder:

```powershell
New-Item -ItemType Directory -Force data/raw
```

Place the CSV at `data/raw/water_potability.csv`. If you have not initialized the project yet, continue from section 4; initialization will copy it into Docker storage.

If Airflow is already running, pause the DAG and wait for active tasks to finish. Copy the CSV into its existing storage:

```powershell
docker compose cp ./data/raw/water_potability.csv airflow-scheduler:/opt/project/data/raw/water_potability.csv
docker compose exec --user 0 airflow-scheduler chown 50000:0 /opt/project/data/raw/water_potability.csv
```

In Airflow, clear the failed `load_data` task and its downstream tasks to retry the failed run, or trigger a new run after the previous one has finished. Unpause the DAG to resume scheduled runs. The same copy commands can replace an existing dataset after active runs finish.

### Docker socket permissions on Linux

On a Linux host, `DOCKER_GID=0` may not match the socket's group. In a Linux terminal, run `stat -c '%g' /var/run/docker.sock`, set `DOCKER_GID` in `.env` to that number, and recreate the scheduler after active tasks finish. Windows Docker Desktop normally uses the default `0`.

## 11. Demonstration checklist

- Show the Airflow graph with all four tasks successful.
- Show a second automatic run after the configured interval.
- Open the matching MLflow run and show its metrics and model artifacts.
- Show both serving containers healthy and a prediction in the web app.
- Match the run ID displayed in the app with the deployed MLflow run.

If submitting the assignment, share the repository link and the required evidence. Generated model files do not need to be committed: the pipeline creates them. Keep `.env`, virtual environments, credentials, and runtime files out of Git.

## Verification status

The Airflow 3 configuration has passed Compose validation and the DAG has passed a Python syntax check. After correcting the deployment prediction check's indentation, a live scheduled Docker run completed all four tasks successfully and wrote a healthy deployment report for the new model. The first-launch checks above explain how to confirm startup, task completion, deployment, and a browser prediction on your machine.
