# Lab 4 - Orchestrating the App with Docker Compose

## Implementation

I containerized the Food-11 application using three Docker Compose services:

- **mlflow:** MLflow Tracking Server and Model Registry, available on port 5000.
- **inference:** FastAPI service loading the registered Food-11 model.
- **frontend:** Streamlit interface, available on port 8501.

The services communicate over the Docker Compose network.

The registered model is accessed using `models:/food11@champion`. I used the MLflow model alias `champion` instead of the legacy `Staging` stage.

MLflow uses a persistent Docker volume named `food11-mlflow-data`.

## Question 1

**What happens without a volume mounted at `/mlflow-data`?**

Without a volume, MLflow stores its database and artifacts in the container's writable filesystem. Stopping and starting the same container preserves the files, but removing that container deletes its writable layer. Creating a new container from the image starts with a new database and no previously registered models.

## Question 2

**Why use a named volume instead of a bind mount?**

A named volume is managed by Docker and is independent of the container lifecycle. It is more portable across host environments and avoids depending on a specific directory in the repository.

A bind mount would also work and can be convenient during development, but it depends on host filesystem paths and permissions.

## Question 3

**Why does `http://mlflow:5000` resolve inside Docker Compose?**

Docker Compose creates a private network with built-in DNS. Containers on that network can communicate using their service names.

Therefore, the inference container can reach the MLflow container using `http://mlflow:5000` without accessing the host machine.

## Question 4

**Why is `INFERENCE_URL` an environment variable?**

Using an environment variable allows the frontend to run in different environments without modifying its code.

Inside Docker Compose, the URL is `http://inference:8000`. Outside Compose, it can be configured to another reachable address, such as `http://127.0.0.1:8000` when the API is exposed locally.

## Question 5

**Why is the inference port not published?**

Only MLflow and Streamlit need to be accessed directly from the host computer.

The inference service is accessed internally by Streamlit using `http://inference:8000` over the Compose network. Therefore, publishing port 8000 to the host is unnecessary.

## Question 6

**What happens if inference starts before MLflow is ready?**

The `depends_on` setting controls startup order but does not guarantee application readiness.

If the inference service attempts to load its model before MLflow is ready, its startup can fail. In our implementation, `restart: on-failure` allows Docker to retry the inference container.

A more robust solution would add a health check to MLflow and make inference depend on its healthy status.

## Question 7

**Which services publish ports?**

The output of `docker compose ps` confirmed:

| Service | Host port | Container port |
| --- | --- | --- |
| mlflow | 5000 | 5000 |
| frontend | 8501 | 8501 |
| inference | Not published | 8000 |

This matches the Docker Compose configuration.

## Question 8

**Does changing the registered model immediately change predictions?**

No. The FastAPI application loads its model into memory during startup. Changing the registered model alias does not automatically reload the running process.

I registered version 2 of `food11`, promoted it to the `champion` alias, and executed:

```bash
docker compose restart inference
```

The inference logs confirmed that the model loaded successfully after restarting.

For this experiment, versions 1 and 2 reference the same trained model, so their predictions are identical.

## Question 9

**Why does restarting inference work without rebuilding its image?**

The inference Docker image contains the application code and dependencies. The registered model is loaded from MLflow when the application starts.

Restarting the container runs the startup code again and resolves the current `champion` alias. Therefore, changing the model version does not require rebuilding the image.

## Question 10

**What happens after `docker compose down`, and what changes with `down -v`?**

I tested:

```bash
docker compose down
docker compose up -d --no-build
```

After recreating the containers, MLflow still contained the registered model and its champion alias.

This confirms that the Docker volume persisted after container removal.

Normally, `docker compose down -v` removes non-external named volumes declared by the Compose project.

However, my configuration declares `food11-mlflow-data` as an external volume, which is not removed by `docker compose down -v`.

I preserved this volume because it contains the trained model and the MLflow database.

## Question 11

**What would be required to scale inference or make MLflow highly available?**

Multiple inference replicas would require a load balancer or reverse proxy to distribute requests.

For multi-machine deployment and fault tolerance, an orchestrator such as Kubernetes could manage scheduling, scaling, and automatic recovery.

MLflow would also require a highly available external database and shared artifact storage. Docker Compose alone does not provide automatic multi-host failover.

## End-to-End Verification

The Streamlit frontend successfully sent a Food-11 evaluation image to FastAPI and displayed a prediction.

- Actual category: Bread
- Predicted category: Vegetable-Fruit
- Confidence: approximately 87.0%
- Application integration: Successful
- Classification: Incorrect

The incorrect classification reflects model accuracy, not a deployment failure.

After promoting version 2 to `champion` and restarting inference, the logs confirmed:

```text
Loading model: models:/food11@champion
Model loaded successfully!
INFO: Application startup complete.
```

## Conclusion

Docker Compose successfully orchestrated MLflow, FastAPI, and Streamlit. The application supports image uploads, inference, persistent model storage, and model version changes without rebuilding the inference image.

## Follow-up audit — safe volume removal experiment

**PASS** — On disposable volume: down/up preserved file; down -v/up removed it

The destructive `down -v` test was run ONLY on an isolated temporary Compose project and its temporary volume, never on the existing `food11-mlflow-data` model volume. The real project's volume is declared `external: true`, and Compose does not remove such an external volume through its `down -v` command.
