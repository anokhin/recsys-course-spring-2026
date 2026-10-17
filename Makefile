SEED       ?= 31312
EPISODES   ?= 30000
DATA_DIR   ?= ./data

VENV   = .venv
PYTHON = $(VENV)/bin/python
PIP    = $(VENV)/bin/pip

# Локально часто есть только docker-compose, в CI — docker compose
DOCKER_COMPOSE ?= $(shell docker compose version >/dev/null 2>&1 && echo "docker compose" || echo "docker-compose")

.PHONY: setup run clean

setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip --timeout 120 -q
	$(PIP) install -r sim/requirements.txt --timeout 120 -q
	$(PIP) install -r botify/requirements.txt --timeout 120 -q
	cd botify && $(DOCKER_COMPOSE) down -v --remove-orphans 2>/dev/null || true
	cd botify && $(DOCKER_COMPOSE) up -d --build --force-recreate --scale recommender=2
	sleep 20

run:
	cd sim && echo "n" | ../$(PYTHON) -m sim.run \
		--episodes $(EPISODES) \
		--config   config/env.yml \
		single --recommender remote --seed $(SEED)
	mkdir -p $(DATA_DIR)
	$(PYTHON) script/dataclient.py --recommender 2 log2local $(DATA_DIR)
	$(PYTHON) analyze_ab.py --data $(DATA_DIR) --output $(DATA_DIR)/ab_result.json

clean:
	cd botify && $(DOCKER_COMPOSE) down -v --remove-orphans 2>/dev/null || true
	rm -rf $(VENV)
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
