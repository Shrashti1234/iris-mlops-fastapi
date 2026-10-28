import json,pathlib
r=pathlib.Path('iris-mlops'); (r/'notebooks').mkdir(exist_ok=True)
cells=[]
def md(s): cells.append({'cell_type':'markdown','metadata':{},'source':s.splitlines(True)})
def code(s): cells.append({'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':s.splitlines(True)})
md('# Iris Species Classification — Kaggle to Docker\nCase study: predict Iris-setosa, Iris-versicolor or Iris-virginica from four flower measurements in cm. Dataset: https://www.kaggle.com/datasets/uciml/iris (original Fisher/UCI Iris). Training and evaluation run in this Jupyter notebook, not a VS Code training script.\n\nIn Colab choose Runtime → Run all. The final cell downloads model.joblib and metrics.json. A stratified 80/20 holdout is kept separate from training. Scaling is fitted only on the training split.')
code('%pip install -q scikit-learn==1.7.2 pandas==2.3.3 joblib==1.5.2')
code('''from pathlib import Path
import io, zipfile, urllib.request, json, hashlib, platform
import pandas as pd
import sklearn, joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
(ROOT / "data").mkdir(exist_ok=True)
(ROOT / "artifacts").mkdir(exist_ok=True)
csv_path = ROOT / "data" / "Iris.csv"
if not csv_path.exists():
    raw = urllib.request.urlopen("https://www.kaggle.com/api/v1/datasets/download/uciml/iris", timeout=60).read()
    archive = zipfile.ZipFile(io.BytesIO(raw))
    csv_name = next(n for n in archive.namelist() if n.lower().endswith("iris.csv"))
    csv_path.write_bytes(archive.read(csv_name))
df = pd.read_csv(csv_path)
print("Shape:", df.shape)
print(df.head().to_string(index=False))
print("Class counts:", df.Species.value_counts().to_dict())
print("Missing values:", int(df.isna().sum().sum()))''')
code('''features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
X = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]].copy()
X.columns = features
y = df["Species"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
model = Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression(max_iter=1000, random_state=42))])
model.fit(X_train, y_train)
predictions = model.predict(X_test)
metrics = {"accuracy": float(accuracy_score(y_test, predictions)), "macro_f1": float(f1_score(y_test, predictions, average="macro")), "train_samples": len(X_train), "test_samples": len(X_test), "random_state": 42}
print(json.dumps(metrics, indent=2))
print(classification_report(y_test, predictions))
print("Confusion matrix (classes in alphabetical order):")
print(confusion_matrix(y_test, predictions))
assert metrics["accuracy"] >= 0.85''')
code('''parameters = model.named_steps["classifier"].get_params()
bundle = {"model": model, "parameters": parameters, "features": features, "classes": model.classes_.tolist(), "metrics": metrics,
          "sklearn_version": sklearn.__version__, "dataset": "uciml/iris", "dataset_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest()}
joblib.dump(bundle, ROOT / "artifacts" / "model.joblib")
(ROOT / "artifacts" / "metrics.json").write_text(json.dumps(metrics, indent=2))
(ROOT / "artifacts" / "parameters.json").write_text(json.dumps(parameters, indent=2))
loaded = joblib.load(ROOT / "artifacts" / "model.joblib")
assert (loaded["model"].predict(X_test) == predictions).all()
print("Saved model and parameters; reload predictions match all test predictions.")
print("Example:", loaded["model"].predict(pd.DataFrame([[5.1,3.5,1.4,0.2]], columns=features))[0])''')
code('''try:
    from google.colab import files
except ImportError:
    print("Jupyter execution complete. Artifacts:", ROOT / "artifacts")
else:
    files.download(str(ROOT / "artifacts" / "model.joblib"))
    files.download(str(ROOT / "artifacts" / "metrics.json"))''')
nb={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'},'colab':{'name':'Iris_Training.ipynb'}},'nbformat':4,'nbformat_minor':5}
for i,c in enumerate(cells): c['id']=f'cell-{i}'
(r/'notebooks'/'Iris_Training.ipynb').write_text(json.dumps(nb,indent=2),encoding='utf-8')

