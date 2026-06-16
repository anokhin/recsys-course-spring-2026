## Homework 2 Report

### Abstract

Предложен рекомендер, использующий эмбеддинги llama3.1 — той же модели, на которой симулятор генерирует описание треков. Для каждого трека оффлайн предвычислены top-50 ближайших соседей в эмбеддинг-пространстве (FAISS IndexFlatIP). На этапе сервинга кандидаты собираются из двух I2I-источников: embedding-based (основной, использует семантическую близость треков) и SasRec (дополнительный, коллаборативный сигнал). Финальный выбор — взвешенный скоринг: кандидаты из обоих источников получают бонус, повтор артиста в сессии штрафуется. Контроль — SasRec-I2I, тритмент — ML-реранкер на эмбеддингах.

### Детали реализации

**Архитектура:**

```
User Request (POST /next/<user>)
  ├── [Control, 50%]  SasRec-I2I Recommender ──→ track
  └── [Treatment, 50%] Embedding I2I + Artist Diversity
        ├── Load user history (Redis listen_history, DB 2)
        ├── Count artist occurrences in session
        ├── For last 3 tracks in history:
        │     ├── Get embedding I2I candidates (Redis DB 6)
        │     └── Get SasRec I2I candidates  (Redis DB 4)
        ├── Merge, deduplicate, remove already-seen tracks
        ├── Score each candidate:
        │     score = 3.0 × [in_emb_i2i]
        │           + 2.0 × [in_sas_i2i]
        │           + 1.5 × [in_both]
        │           − 0.4 × [artist_session_count]
        └── Return max-score candidate (fallback: RandomRecommender)
```

**Генерация embedding I2I (оффлайн):**
1. Загружены эмбеддинги llama3.1 из `sim/data/embeddings.npy` (16198 × 4096, float32)
2. Построен FAISS IndexFlatIP — inner product на нормализованных векторах эквивалентен cosine similarity
3. Для каждого из 16198 треков найден top-50 ближайших соседей
4. Результат сохранён в `botify/data/embedding_i2i.jsonl` (5.5 MB, формат идентичен `sasrec_i2i.jsonl`)

**Файлы:**
- `botify/botify/recommenders/ml_reranker.py` — класс MLReranker (скоринг + artist diversity)
- `botify/data/embedding_i2i.jsonl` — embedding-based I2I рекомендации
- `botify/botify/config.json` — Redis DB 6 для embedding I2I
- `botify/botify/server.py` — загрузка embedding I2I в Redis, инициализация MLReranker
- `botify/botify/experiment.py` — эксперимент `ML_RERANKER` (Split.HALF_HALF, Control=SasRec-I2I)

### Результаты A/B эксперимента

| Метрика | Контроль (C) | Тритмент (T1) | Effect, % | CI lower, % | CI upper, % | Значимо |
|---------|-------------|---------------|-----------|-------------|-------------|---------|
| mean_time_per_session | * | * | * | * | * | * |
| mean_tracks_per_session | * | * | * | * | * | * |
| mean_request_latency | * | * | * | * | * | * |

> Заполнить после `make run` из `data/ab_result.json`.
