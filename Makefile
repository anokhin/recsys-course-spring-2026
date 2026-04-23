SEED       ?= 31312
EPISODES   ?= 30000
DATA_DIR   ?= ./data

VENV   = .venv
PYTHON = $(VENV)/bin/python
PIP    = $(VENV)/bin/pip

.PHONY: setup run update_model clean

setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip --timeout 120 -q
	$(PIP) install -r sim/requirements.txt --timeout 120 -q
	$(PIP) install -r botify/requirements.txt --timeout 120 -q
	cd botify && docker compose down -v --remove-orphans 2>/dev/null || true
	cd botify && docker compose up -d --build --force-recreate --scale recommender=2
	sleep 20

run:
	cd sim && echo "n" | ../$(PYTHON) -m sim.run \
		--episodes $(EPISODES) \
		--config   config/env.yml \
		single --recommender remote --seed $(SEED)
	mkdir -p $(DATA_DIR)
	$(PYTHON) script/dataclient.py --recommender 2 log2local $(DATA_DIR)
	$(PYTHON) analyze_ab.py --data $(DATA_DIR) --output $(DATA_DIR)/ab_result.json

update_model:
	$(PIP) install -r requirements-training.txt --timeout 120 -q
	$(PYTHON) script/train_two_tower.py \
		--logs $(DATA_DIR) \
		--tracks botify/data/tracks.json \
		--output botify/data/two_tower_candidates.jsonl \
		--meta botify/data/two_tower_meta.pkl \
		--topk 500 --epochs 30 --seed $(SEED)
	$(PYTHON) script/train_lambdarank.py \
		--logs $(DATA_DIR) \
		--tracks botify/data/tracks.json \
		--candidates botify/data/two_tower_candidates.jsonl \
		--meta botify/data/two_tower_meta.pkl \
		--output botify/data/learned_i2i.jsonl \
		--topk 200 --seed $(SEED)
	cd botify && docker compose up -d --build --force-recreate --scale recommender=2

clean:
	cd botify && docker compose down -v --remove-orphans 2>/dev/null || true
	rm -rf $(VENV)
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
