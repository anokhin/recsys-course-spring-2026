SEED       ?= 31312
EPISODES   ?= 30000
DATA_DIR   ?= ./data

VENV   = .venv
PYTHON = $(VENV)/bin/python
PIP    = $(VENV)/bin/pip

.PHONY: setup run clean

setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip --timeout 120 -q
	$(PIP) install -r sim/requirements.txt --timeout 120 -q
	$(PIP) install -r botify/requirements.txt --timeout 120 -q
	cd botify && docker compose down -v --remove-orphans 2>/dev/null || true
	cd botify && docker compose up -d --build --force-recreate --scale recommender=2
	for i in 1 2 3 4 5 6; do \
		sleep 10; \
		curl -sf http://localhost:5001/ && break; \
		echo "Attempt $$i/6..."; \
	done
	curl -sf http://localhost:5001/ || { cd botify && docker compose logs recommender | grep -v "gunicorn\|arbiter\|SIGTERM\|Shutting\|HaltServer\|Worker exiting" | tail -60; exit 1; }

run:
	cd sim && echo "n" | ../$(PYTHON) -m sim.run \
		--episodes $(EPISODES) \
		--config   config/env.yml \
		single --recommender remote --seed $(SEED)
	mkdir -p $(DATA_DIR)
	$(PYTHON) script/dataclient.py --recommender 2 log2local $(DATA_DIR)
	$(PYTHON) analyze_ab.py --data $(DATA_DIR) --output $(DATA_DIR)/ab_result.json

clean:
	cd botify && docker compose down -v --remove-orphans 2>/dev/null || true
	rm -rf $(VENV)
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
