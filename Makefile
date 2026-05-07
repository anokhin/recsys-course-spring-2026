SEED          ?= 31312
EPISODES      ?= 30000
DATA_DIR      ?= ./data
TRAIN_DIR     ?= ./data/train
RETRIEVAL_K   ?= 500
RERANK_K      ?= 200

VENV   = .venv
PYTHON = $(VENV)/bin/python
PIP    = $(VENV)/bin/pip

.PHONY: setup run clean collect_data update_model

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

collect_data:
	cd sim && echo "n" | ../$(PYTHON) -m sim.run \
		--episodes $(EPISODES) \
		--config   config/env.yml \
		single --recommender remote --seed $(SEED)
	mkdir -p $(TRAIN_DIR)
	$(PYTHON) script/dataclient.py --recommender 2 log2local $(TRAIN_DIR)

update_model:
	$(PIP) install -r requirements-training.txt --timeout 240 -q
	$(PYTHON) script/train_retrieval.py \
		--logs $(TRAIN_DIR) \
		--tracks botify/data/tracks.json \
		--cand-out botify/data/cand_pool.jsonl \
		--emb-out botify/data/item_embs.npz \
		--topk $(RETRIEVAL_K) \
		--seed $(SEED)
	$(PYTHON) script/train_survival.py \
		--logs $(TRAIN_DIR) \
		--candidates botify/data/cand_pool.jsonl \
		--embeddings botify/data/item_embs.npz \
		--tracks botify/data/tracks.json \
		--out botify/data/learned_i2i.jsonl \
		--topk $(RERANK_K) \
		--seed $(SEED)
	cd botify && docker compose up -d --build --force-recreate --scale recommender=2
	sleep 20

clean:
	cd botify && docker compose down -v --remove-orphans 2>/dev/null || true
	rm -rf $(VENV)
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
