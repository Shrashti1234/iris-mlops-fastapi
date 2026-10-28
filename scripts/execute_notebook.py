from pathlib import Path
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
path = root / 'notebooks' / 'Iris_Training.ipynb'
notebook = nbformat.read(path, as_version=4)
client = NotebookClient(notebook, timeout=600, kernel_name='iris-mlops', resources={'metadata': {'path': str(root)}})
try:
    client.execute()
finally:
    nbformat.write(notebook, path)
for cell in notebook.cells:
    for output in cell.get('outputs', []):
        if output.get('output_type') == 'stream':
            print(output.get('text', ''))
