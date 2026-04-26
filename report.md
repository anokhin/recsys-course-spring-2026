# Отчёт по ДЗ-2

## 1. Архитектура

```mermaid
flowchart LR

    %% Источники данных
    A[Статические даннные<br/>tracks / sasrec_i2i / lightfm_i2i]
    B[История<br/>(Redis)]

    %% Pipeline Обучения
    A --> C[Docker build → train.py]
    C --> D[SVD (SasRec ⊕ LightFM)]
    D --> E[64D item embeddings]

    %% Inference pipeline
    B --> F[Якоря сессии<br/>(up to 5)]
    F --> G[Генерация кандидатов<br/>SasRec ∪ LightFM<br/>(top-30 per anchor)]

    %% Merge
    E --> H
    G --> H

    %% Scoring
    H[Scoring]
    H --> I[score(c) = cos(intent, e_c)<br/>− λ · artist_count(c)<br/>+ β · agree(c)]

    %% Output
    I --> J[Топ-рекомендация]
```

### 1.1. Оффлайн: эмбеддинги треков

Скрипт `botify.ml.train` запускается на этапе `docker build`,
поэтому контейнер всегда стартует с одинаковыми артефактами.

1. Читаем `sasrec_i2i.jsonl` и `lightfm_i2i.jsonl`. Для каждой
   пары (якорь, сосед) кладём в разреженную матрицу
   `1 / log2(2 + rank)` — близких соседей считаем тяжелее.
2. Складываем графы с весами `1.0` (SasRec) и `0.6` (LightFM), итоговая матрица: `M = (W + Wᵀ) / 2`.
4. Сохраняем три артефакта в `botify/data/ml/`:
   `track_embeddings.npy`, `track_index.json`, `artist_index.json`.

### 2.2. Онлайн: `MLReranker`
1. **История.** Последние 10 событий из Redis
   (`user:<id>:listens`).
2. **Якоря.** До 5 треков из истории с весом
   `max(time, 0.05) · 0.85^position`. Свежие и долго слушаемые
   важнее.
3. **Кандидаты.** По каждому якорю достаём top-30 из SasRec и
   top-30 из LightFM, объединяем, выкидываем уже услышанное.

## 3. Результаты A/B

Локальный прогон, 30 000 эпизодов, `SEED=31312`. Полные числа
лежат в `data/ab_result.json` (`primary_metric =
mean_time_per_session`).

| Метрика | Контроль | Треатмент | Эффект | 95% ДИ | Значимо |
|---|---|---|---|---|---|
| `mean_time_per_session` | 7.01 | 8.79 | **+25.5%** | [+23.21%, +27.83%] | Да |
| `mean_tracks_per_session` | 11.99 | 13.79 | **+15.0%** | [+13.60%, +16.46%] | Да |
| `mean_request_latency`   | 0.57 | 1.26 | **+122.16%** | [+116.19%, +128.12%] | Да |
