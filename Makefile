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
	@echo "Waiting for botify readiness on http://localhost:5001/ ..."
	@for i in $$(seq 1 90); do \
		if curl -fsS http://localhost:5001/ >/tmp/botify_health.json 2>/dev/null && grep -q '"status"' /tmp/botify_health.json; then \
			echo "Botify is ready"; \
			break; \
		fi; \
		if [ $$i -eq 90 ]; then \
			echo "Botify did not become ready in time"; \
			cd botify && docker compose ps && docker compose logs --tail=200; \
			exit 1; \
		fi; \
		sleep 2; \
	done

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
