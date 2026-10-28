import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

def test_health(client):
    assert client.get('/health').json() == {'status': 'ok', 'model_loaded': True}

@pytest.mark.parametrize('values,species', [
    ([5.1,3.5,1.4,0.2], 'Iris-setosa'),
    ([6.0,2.9,4.5,1.5], 'Iris-versicolor'),
    ([6.7,3.0,5.7,2.5], 'Iris-virginica'),
])
def test_prediction(client, values, species):
    keys = ['sepal_length','sepal_width','petal_length','petal_width']
    response = client.post('/predict', json=dict(zip(keys, values)))
    assert response.status_code == 200
    result = response.json()
    assert result['species'] == species
    assert sum(result['probabilities'].values()) == pytest.approx(1)

@pytest.mark.parametrize('body', [{}, {'sepal_length': -1,'sepal_width':3,'petal_length':2,'petal_width':1},
    {'sepal_length':5,'sepal_width':3,'petal_length':2,'petal_width':1,'unexpected':1}])
def test_invalid_input(client, body):
    assert client.post('/predict', json=body).status_code == 422

def test_quality_and_parameters(client):
    info = client.get('/model-info').json()
    assert info['metrics']['accuracy'] >= 0.85
    assert info['parameters']['max_iter'] == 1000
