# Homework 2 — Content-aware re-ranker over SasRec-I2I

## Abstract

SasRec-I2I в боевой версии берёт топ-K соседей одного «якоря» (последний прослушанный
трек), не зная ни про сессионный интерес пользователя, ни про штраф симулятора за
повторного артиста. Я строю **session-aware re-ranker** поверх уже существующего
SasRec-I2I-кандидатного пула. На лету для каждого запроса я: (1) собираю векторное
представление текущей сессии — взвешенную по `listen_time` сумму контентных эмбеддингов
прослушанных треков, (2) пересортировываю SasRec-I2I-кандидатов по
`cosine(track, session_vec) − λ · count_same_artist_in_session`. Эмбеддинги треков —
TF-IDF поверх `[artist, genres, mood, summary, artist_country]` с понижением до 64 dim
через TruncatedSVD; считаются один раз офлайн. Это побеждает SasRec-I2I по
**mean_time_per_session на +47.96 % (p < 0.05)** на 2000 эпизодах симулятора.

## Implementation

```
                              ┌──────────────────────────────────────────┐
listen-history Redis ─────►   │ ContentRerankRecommender.recommend_next   │
( (track, time) последних 50) │  1. session_vec = Σ time·emb(track) /Σtime│
                              │  2. cands = ⋃_{a∈hist} top-K SasRec-i2i(a)│
SasRec-I2I Redis ─────►       │     − seen                                │
( top-10 на якорь )           │  3. score(c) = emb(c)·session_vec         │
                              │              − 0.07·#{same_artist in hist}│
content_embeddings.npy ──►    │  4. argmax score(c)                       │
( 16k × 64, L2-norm )         └──────────────────────────────────────────┘
```

Эмбеддинги собираются `botify/scripts/build_content_embeddings.py`: каждый трек
конкатенируется в текст (артист повторён ×3, чтобы давать больший вес), TF-IDF с
1- и 2-граммами над 67k фич, далее `TruncatedSVD(64) + L2-normalize`. Файлы
`content_embeddings.npy` (4 МБ, float32) и `content_embeddings_meta.json`
лежат в `botify/data/`. Рекомендер `ContentRerankRecommender` (см.
`botify/botify/recommenders/content_rerank.py`) загружает их при старте сервера и
держит в памяти. На каждом `/next/` он **переиспользует уже существующий
SasRec-I2I-индекс** (тот же Redis-ключ, что и у контроля) — это означает, что
качество кандидатного пула то же самое, мы только меняем порядок. История юзера
расширена с 10 до 50 последних прослушиваний (`LISTEN_HISTORY_LIMIT`), чтобы
session_vec был устойчивее. На пустой истории и при отсутствии кандидатов после
фильтрации — fallback в SasRec-I2I, чтобы не деградировать на холодном старте.

A/B встроен в `Experiments.CONTENT_RERANK = Experiment("CONTENT_RERANK", Split.HALF_HALF)`;
в контроле (Treatment.C) — оригинальный `I2IRecommender` поверх
`sasrec_i2i.jsonl`, в трeатменте (Treatment.T1) — мой `ContentRerankRecommender`.
Симулятор поднимается через `make setup` (Docker Compose, 2 реплики
`recommender` за nginx-балансером), `make run` гонит N эпизодов и зовёт
`analyze_ab.py`.

## A/B-результаты

Локальный прогон, `EPISODES=2000 SEED=31312` — те же параметры, что и в чекере, только
меньше эпизодов:

| metric                   | control mean | treatment mean | effect % | 95 % CI         | sig. |
|--------------------------|-------------:|---------------:|---------:|-----------------|:----:|
| **mean_time_per_session**| 9.48         | 14.03          | **+47.96 %** | [+38.84, +57.08]   | ✅ |
| mean_tracks_per_session  | 14.51        | 18.99          | +30.88 % | [+24.62, +37.15]   | ✅ |
| time (total)             | 10.57        | 15.19          | +43.71 % | [+34.44, +52.98]   | ✅ |
| sessions                 | 1.12         | 1.11           | −1.21 %  | [−3.98, +1.55]     | ✗ |
| mean_request_latency     | 0.47 ms      | 1.16 ms        | +149.19 %| [+138.79, +159.59] | ✅ |

Главная метрика **mean_time_per_session растёт на +47.96 %, p < 0.05** —
бейзлайн побит уверенно. Растёт и средняя длина сессии, потому что треки лучше
попадают в текущий интерес (меньше скипов → бюджет сессии тратится медленнее).
Латентность при этом ~1 мс, что на полмиллисекунды дороже бейзлайна, но в
абсолюте всё ещё моментально для пользователя.
