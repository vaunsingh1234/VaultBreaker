.PHONY: setup data train eval test app api fast clean

PYTHON := python3

setup:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e . --no-deps

data:
	$(PYTHON) scripts/make_dataset.py

features:
	$(PYTHON) scripts/extract_features.py --num-workers 4

train:
	$(PYTHON) scripts/train.py

eval:
	$(PYTHON) scripts/evaluate.py

fast:
	$(PYTHON) scripts/make_dataset.py --fast
	$(PYTHON) scripts/extract_features.py --num-workers 2
	$(PYTHON) scripts/train.py --fast
	$(PYTHON) scripts/evaluate.py

test:
	pytest tests/ -v

app:
	streamlit run app/streamlit_app.py --server.port 8501 --server.headless true

api:
	uvicorn api.main:app --host 0.0.0.0 --port 8000

clean:
	rm -rf __pycache__ .pytest_cache data/features/cache *.egg-info build dist
