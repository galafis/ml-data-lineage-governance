.PHONY: install test lint run clean docker-build docker-up docker-down

PYTHON ?= python
PIP ?= pip

install:
	$(PIP) install -r requirements.txt

test:
	$(PYTHON) -m pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

lint:
	$(PYTHON) -m py_compile main.py
	$(PYTHON) -m py_compile src/lineage/tracker.py
	$(PYTHON) -m py_compile src/lineage/graph.py
	$(PYTHON) -m py_compile src/quality/validator.py
	$(PYTHON) -m py_compile src/quality/profiler.py
	$(PYTHON) -m py_compile src/governance/compliance.py
	$(PYTHON) -m py_compile src/governance/catalog.py
	$(PYTHON) -m py_compile src/schema/evolution.py

run:
	$(PYTHON) main.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage coverage.xml

docker-build:
	docker-compose -f docker/docker-compose.yml build

docker-up:
	docker-compose -f docker/docker-compose.yml up -d

docker-down:
	docker-compose -f docker/docker-compose.yml down
