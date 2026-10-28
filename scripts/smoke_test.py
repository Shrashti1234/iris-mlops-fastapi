import json
import time
import urllib.request
import urllib.error
from http.client import HTTPException

base = 'http://localhost:8000'
for attempt in range(60):
    try:
        with urllib.request.urlopen(base+'/health', timeout=2) as response:
            assert json.load(response)['model_loaded']
        break
    except (OSError, HTTPException):
        time.sleep(1)
else:
    raise RuntimeError('API did not become ready')
payload = dict(sepal_length=5.1,sepal_width=3.5,petal_length=1.4,petal_width=0.2)
request = urllib.request.Request(base+'/predict', data=json.dumps(payload).encode(), headers={'Content-Type':'application/json'})
with urllib.request.urlopen(request) as response:
    result = json.load(response)
assert result['species'] == 'Iris-setosa', result
print(json.dumps(result, indent=2))
with urllib.request.urlopen(base+'/model-info') as response:
    assert json.load(response)['metrics']['accuracy'] >= 0.85
with urllib.request.urlopen(base+'/openapi.json') as response:
    assert '/predict' in json.load(response)['paths']
print('Docker API smoke test passed')
