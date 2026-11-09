# Lab 3 - Containerizing the Model with Docker

## Question 1 - Model Registration

I registered my Food-11 ResNet18 model in MLflow under the name `food11`. MLflow assigned it version 1, and I assigned the alias `champion`.

The model came from my new training run:
- Run name: painted-flea-294
- Run ID: cebda5e167534b948c65e211309a56b6
- Final validation accuracy: 0.5830
- Test accuracy: 0.6369

A logged model is associated with a particular training run. A registered model has its own name and versions, allowing different training runs to produce models that can be managed under one name.

## Question 2 - Aliases and Versions

MLflow model versions allow multiple trained models to exist under the same registered name.

Aliases such as `champion` provide a movable reference to a particular version. This is more flexible than the older fixed stage system because the alias can be reassigned without changing the serving code.

My registered model is `food11`, version 1, with alias `champion`.

## Question 3 - Loading from the Registry

My FastAPI application uses:

`mlflow.pyfunc.load_model("models:/food11@champion")`

This avoids hardcoding a local `.pth` file path or a particular training run ID.

If I register a better version later, I can move the `champion` alias to that version. The application can then load the newer model on its next startup without changing the Python source code.

## Question 4 - Docker Layer Caching

My Dockerfile uses a multi-stage build.

In the builder stage, I copy `pyproject.toml` and `uv.lock` before installing the dependencies with `uv sync --frozen --no-dev --no-install-project`.

I copy the application source code afterward, in the runtime stage.

This ordering allows Docker to reuse its dependency layer when only the application source changes. This is important in my project because downloading PyTorch, Torchvision, and MLflow during the initial Docker build took a long time.

## Question 5 - Image Size and History

I successfully built the multi-stage image `food11-api:latest`.

Docker reported approximately:
- Image disk usage: 2 GB
- Content size: 428 MB
- Virtual environment layer: 1.44 GB
- Application source layer: 53.2 KB
- Additional `libgomp1` layer: 848 KB

I inspected the image using:

`docker history food11-api:latest`

The virtual environment was by far the largest component, mainly because of machine-learning dependencies.

A naive single-stage build would generally retain additional builder tools and potentially caches. The multi-stage approach keeps the final runtime image separate from the dependency-building environment.

I did not build a separate naive image, so I cannot claim a measured size difference between the two approaches.

## Question 6 - Dockerignore

I created a `.dockerignore` file to exclude unnecessary local files, including `.venv`, `data`, `mlruns`, `mlflow.db`, `.git`, and Python cache files.

My Docker build context was only approximately 564 KB, despite having a large Food-11 ZIP file and local model artifacts in my project.

This reduces unnecessary context transfer and avoids accidentally copying machine-specific files.

Excluding the Windows `.venv` directory is especially important because the Docker container needs Linux-compatible Python packages.

## Question 7 - Docker Networking

Inside a Docker container, `127.0.0.1` refers to the container itself, not my Windows computer.

I configured the container to use:

`MLFLOW_TRACKING_URI=http://host.docker.internal:5000`

This hostname allows Docker Desktop containers to contact services running on the Windows host.

Initially, MLflow returned a 403 error due to its hostname security validation. I restarted MLflow with an appropriate `--allowed-hosts` configuration and verified that requests using `host.docker.internal:5000` returned HTTP 200.

## Question 8 - Restarting the Same Image

I successfully started my API using the previously built Docker image, without rebuilding it.

The model is loaded during application startup using the `champion` alias. This separates the application image from the choice of model version.

However, my registered model's artifacts were stored at a Windows `file:` URI. The container could reach the MLflow server but could not access those Windows files directly.

I solved this for local testing by bind-mounting the `mlruns` directory into the container at the path expected by MLflow. The mount needed write access because MLflow generated a registered-model metadata file during loading.

A production deployment would need a portable artifact-storage solution instead of relying on this Windows-specific mount.

I confirmed that the same image could start successfully with the correct runtime configuration. I did not test switching the alias to a different model version.

## Question 9 - Preparing for CI/CD and Kubernetes

My image `food11-api:latest` currently exists only in my local Docker environment.

To deploy it through CI/CD or Kubernetes, I would need to publish it to a container registry such as Docker Hub or GitHub Container Registry.

I should use a versioned image tag and preferably an immutable digest for reproducible deployments.

The deployment would also require a reachable MLflow tracking server, accessible model artifact storage, appropriate configuration and credentials, and a build pipeline that produces the image from the Git repository.

The current Windows bind-mount workaround is suitable for my local lab but is not portable to a Kubernetes cluster.

## Local Verification

- MLflow registered model: `food11`, version 1, alias `champion`
- Docker image: `food11-api:latest`
- Dockerized `GET /health`: HTTP 200, `{"status":"ok"}`
- Dockerized `POST /predict`: successful response
- Sample predicted category: `Vegetable-Fruit`
- Sample confidence: 0.8702

The prediction was made using an image from the Bread evaluation folder. The category was incorrect, but the test verified that the container successfully loaded the model and returned a prediction.

The model achieved 0.6369 test accuracy during training.
