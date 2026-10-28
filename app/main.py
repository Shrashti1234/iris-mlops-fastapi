from contextlib import asynccontextmanager
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field, ConfigDict

MODEL_PATH = Path(__file__).resolve().parents[1] / 'artifacts' / 'model.joblib'

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.bundle = joblib.load(MODEL_PATH)
    yield

app = FastAPI(title='Iris Species Prediction API', version='1.0.0', lifespan=lifespan)

class Measurements(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    sepal_length: float = Field(gt=0, le=20, examples=[5.1])
    sepal_width: float = Field(gt=0, le=20, examples=[3.5])
    petal_length: float = Field(gt=0, le=20, examples=[1.4])
    petal_width: float = Field(gt=0, le=20, examples=[0.2])

@app.get('/health')
def health():
    return {'status': 'ok', 'model_loaded': hasattr(app.state, 'bundle')}

@app.get('/model-info')
def model_info():
    b = app.state.bundle
    return {'algorithm': 'StandardScaler + LogisticRegression', 'features': b['features'],
            'classes': b['classes'], 'metrics': b['metrics'], 'parameters': b['parameters']}

@app.post('/predict')
def predict(data: Measurements):
    b = app.state.bundle
    frame = pd.DataFrame([data.model_dump()], columns=b['features'])
    probabilities = b['model'].predict_proba(frame)[0]
    return {'species': str(b['model'].predict(frame)[0]),
            'confidence': float(max(probabilities)),
            'probabilities': dict(zip(b['classes'], map(float, probabilities)))}
