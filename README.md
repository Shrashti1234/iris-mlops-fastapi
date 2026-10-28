# Iris MLOps: Kaggle → Jupyter/Colab → FastAPI → Docker → GitHub Actions

[![ML API CI](https://github.com/Shrashti1234/iris-mlops-fastapi/actions/workflows/ci.yml/badge.svg)](https://github.com/Shrashti1234/iris-mlops-fastapi/actions/workflows/ci.yml)

[Open training in Google Colab](https://colab.research.google.com/github/Shrashti1234/iris-mlops-fastapi/blob/main/notebooks/Iris_Training.ipynb)

## Case study

Predict one of three Iris species using sepal length, sepal width, petal length and petal width (centimetres). Source: [Kaggle uciml/iris](https://www.kaggle.com/datasets/uciml/iris), originally Fisher/UCI Iris. The downloaded Kaggle CSV is included in `data/Iris.csv` (150 examples, four predictors). `Id` is excluded. This is an educational classification demonstration.

## 1. Train and test in Jupyter or Colab

Open the Colab link and choose **Runtime → Run all**. Alternatively open `notebooks/Iris_Training.ipynb` in Jupyter. All training code is inside the notebook. A stratified 80/20 split with random seed 42 produces 120 training and 30 test rows. A pipeline fits StandardScaler only on training data, then LogisticRegression. The notebook prints accuracy, macro F1, classification report and confusion matrix on the untouched test set.

The final cells save `artifacts/model.joblib`, `metrics.json` and `parameters.json`, and check that reloaded predictions match. The joblib bundle includes the fitted pipeline (including learned coefficients and scaling parameters), estimator hyperparameters, feature order, class labels, test metrics, library version and dataset hash. In Colab download the artifacts and place them in the local `artifacts/` folder. Use only trusted joblib files.

For automated execution using a real Jupyter kernel:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-notebook.txt
.\.venv\Scripts\python -m ipykernel install --user --name iris-mlops --display-name "Iris MLOps"
.\.venv\Scripts\python scripts/execute_notebook.py
```

The committed notebook retains its outputs. Local Jupyter execution and hosted Google Colab are distinct environments; the Colab link lets you reproduce the same workflow there.

## 2. Prediction API in VS Code

Open this folder in VS Code. Select `.venv` as the interpreter, then use its terminal:

```powershell
.\.venv\Scripts\python -m pip install -r requirements-dev.txt
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python -m uvicorn app.main:app --reload
```

Open [Swagger UI](http://localhost:8000/docs). Endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Model readiness |
| GET | `/model-info` | Features, labels, hyperparameters and test metrics |
| POST | `/predict` | Species, confidence and class probabilities |

Example request (all measurements in cm):

```json
{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}
```

Expected species: `Iris-setosa`. Missing, negative, non-finite or unexpected inputs are rejected with HTTP 422. Confidence is the model's estimated class probability, not a calibrated guarantee.

## 3. Execute endpoints through Docker

Start Docker Desktop with Linux containers, then:

```powershell
docker build -t iris-api .
docker run -d --name iris-api -p 8000:8000 iris-api
python scripts/smoke_test.py
```

The smoke test waits for readiness and calls the running container's health, prediction, model-info and OpenAPI endpoints. Swagger also works at http://localhost:8000/docs.

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict -ContentType 'application/json' -Body '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```

Stop the demonstration with `docker stop iris-api`.

## 4. GitHub Actions

`.github/workflows/ci.yml` runs on push, pull requests and manual dispatch. It installs dependencies, tests the committed notebook-trained artifact and input validation, builds the Docker image, starts a container and calls its HTTP endpoints. Logs are retained in the Actions run. No Kaggle or deployment secrets are needed because the dataset and trained artifact are committed. CI validates the saved model; retraining stays in the notebook.

For submission show: notebook outputs, saved model and parameters, Swagger prediction, running Docker container and the successful Actions run. Exact evaluation results are in `artifacts/metrics.json`.
