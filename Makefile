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
	@echo "=== Waiting for botify /healthcheck ==="
	@for i in $$(seq 1 90); do \
		if curl -sf http://localhost:5001/ > /dev/null 2>&1; then \
			echo "botify is up after $$i*2 seconds"; \
			break; \
		fi; \
		sleep 2; \
	done
	@echo "=== Recommender-1 logs ==="
	docker logs botify-recommender-1 --tail 80 || true
	@echo "=== Recommender-2 logs ==="
	docker logs botify-recommender-2 --tail 80 || true
	@echo "=== docker ps ==="
	docker ps -a
	@sleep 5

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