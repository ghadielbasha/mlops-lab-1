# MLOps Labs 1–5 — Food-11 classifier

Course project implementing dataset versioning, MLflow experiment tracking, ResNet18 model
training, FastAPI inference, Docker/Compose, Streamlit, GitHub Actions CI/CD, model registry
aliases, and a statistical model promotion gate.

## Lab reports

- [Lab 1 — Git and DVC](lab/lab1.md)
- [Lab 2 — Model training and MLflow](lab/lab2.md)
- [Lab 3 — Docker and FastAPI](lab/lab3.md)
- [Lab 4 — Docker Compose](lab/lab4.md)
- [Lab 5 — CI/CD and promotion](lab/lab5.md)

## Quick setup (Windows PowerShell)

```powershell
git clone https://github.com/ghadielbasha/mlops-lab-1.git
cd mlops-lab-1
uv sync --frozen
uv run --no-sync pytest -q
uv run --no-sync ruff check src tests
```

This repo intentionally excludes the full Food-11 data and the MLflow database/model artifacts.
`data.dvc` contains the full data pointer; `dvc pull` requires the configured remote
actually to contain the objects, and permissions if needed. Check the Lab 1 audit
section for the latest fresh-clone retrieval result.

If the remote is unavailable, obtain Food-11 from the original [Kaggle dataset](https://www.kaggle.com/datasets/karakaggle/food11).
Put the downloaded archive at `data/food11.zip` and run:

```powershell
uv run --no-sync python src/food11/prepare_mini_from_zip.py
```

That produces a small development dataset but does **not** reconstruct the complete
DVC-tracked dataset. Dataset validation for leakage requires a cleaned split
(`data/food11_processed_mini_clean` in the original development environment).

## Model training and serving

Start an MLflow tracking server and register a model before starting inference.
For example, if the `mlflow` Compose service has been initialized:

```powershell
docker compose up -d mlflow
uv run --no-sync python src/food11/train.py --dataset mini --epochs 5 --lr 0.01 --batch-size 32 --register-name food11
```

The model registry needs a `champion` alias pointing to a registered version.
Set it in MLflow UI, or with `MlflowClient.set_registered_model_alias`.
Then:

```powershell
docker compose up -d --build
```

- Streamlit: http://127.0.0.1:8501
- MLflow: http://127.0.0.1:5000

The project's `docker-compose.yml` uses an **external** named volume called
`food11-mlflow-data`; create that volume (`docker volume create food11-mlflow-data`)
if it doesn't already exist. A fresh volume does **not** contain the author's
trained model or registered aliases. The inference service therefore requires
model initialization before it can work.

## CI/CD

Workflow: `.github/workflows/ci.yml`

1. `test`: Ruff and pytest (mocked model; no remote MLflow required).
2. `smoke`: Build API Docker image and check `/health` with `SKIP_MODEL=1`.
3. `push`: On successful pushes to `main`, publish immutable SHA-tagged image to
   `ghcr.io/ghadielbasha/mlops-lab-1`.

Pull requests to protected `main` require successful `test` and `smoke` checks.
The quality gate logic is `src/food11/promote.py`; results are documented in Lab 5.

**Reproducibility limitations:** Complete Food-11 data and MLflow model artifact
storage are not embedded in GitHub. The provided code, tests, Docker images and
reports can be reviewed independently; running model-backed predictions on a new
machine additionally requires an accessible dataset and an initialized registry.
