.PHONY: install data run test clean

install:
	pip install -r requirements.txt

data:
	python scripts/download_data.py

run:
	streamlit run dashboard/app.py

test:
	pytest tests/ -v --tb=short

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -f data/processed/*.parquet
