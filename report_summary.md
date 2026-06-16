# Report Summary: Homework 2 Ideas Analysis

## Идеи по категориям (с повторяемостью)

### 1. Гибридные кандидатные источники (6 из 15)
Комбинация нескольких I2I моделей как кандидатов:

| Отчёт | Источники |
|-------|-----------|
| 3 | SasRec-I2I + LightFM-I2I + HSTU |
| 7 | HSTU + SasRec-I2I (RRF) |
| 9 | SasRec-I2I + LightFM-I2I + персональный список |
| 10 | LightFM retrieval + LightGBM reranker |
| 13 | SasRec-I2I + LightFM-I2I + HSTU (RRF) |
| 14 | SasRec-I2I + LightFM-I2I + LightGBM LambdaRank |

**Вывод:** Самый популярный паттерн — объединить SasRec-I2I, LightFM-I2I и HSTU в единый пул кандидатов.

---

### 2. Diversity penalty по артистам (5 из 15)
Штраф за повтор артиста в сессии для борьбы с `artist_discount_gamma`:

| Отчёт | Подход |
|-------|--------|
| 3 | ML reranking с diversity-штрафом |
| 4 | Формула: `0.15 × count^1.3` |
| 6 | Жадный выбор ≤1 трека на артиста (avg 10.0 vs 6.64 у SasRec) |
| 11 | MMR rerank с artist penalty |
| 13 | Мягкий штраф ×0.25 для последних 3 артистов |

**Вывод:** Почти треть решений использует artist diversity как ключевой механизм улучшения.

---

### 3. Content-based эмбеддинги (4 из 15)
Текстовые признаки треков:

| Отчёт | Метод |
|-------|--------|
| 5 | TF-IDF по title, genre, mood, country, summary |
| 6 | sentence-transformer (all-MiniLM-L6-v2) |
| 11 | Item2Vec (gensim skip-gram) + TruncatedSVD |
| 15 | sentence-transformer + iALS |

**Вывод:** Content-признаки используются как дополнение к collaborative filtering.

---

### 4. ML-модели для ранжирования (5 из 15)

| Отчёт | Модель | Цель |
|-------|--------|------|
| 3 | Линейный скорer | Агрегация фичей |
| 6 | LogReg blender | Классификация (позитивы = взаимные соседи) |
| 9 | Табличная сглаженная модель | Предсказание listen_time |
| 10 | LightGBM quantile loss (α=0.7) | Предсказание верхнего квантиля time |
| 14 | LightGBM LambdaRank | Learning-to-rank |

**Вывод:** LambdaRank и quantile regression — наиболее продвинутые подходы.

---

### 5. Transition matrix / Обучение I2I напрямую из логов (2 из 15)

| Отчёт | Подход |
|-------|--------|
| 12 | `M[a,b] = Σ listen_time` для пар (prev_track, track) |
| 15 | Transition matrix + iALS smoothing (α=0.05) |

**Вывод:** Прямое обучение из переходов — простой и эффективный подход.

---

### 6. RRF — Reciprocal Rank Fusion (2 из 15)

| Отчёт | Формула |
|-------|---------|
| 7 | `score = Σ w/(60 + rank)` |
| 13 | RRF по 3 источникам + artist penalty |

**Вывод:** RRF — стандартный способ объединить несколько ranked lists.

---

### 7. Confidence gate / консервативный serving (2 из 15)

| Отчёт | Логика |
|-------|--------|
| 2 | Override только при `best_score > 0.78` и margin > 0.10 |
| 14 | Override только при score margin ≥ 0.05 и artist constraint |

**Вывод:** Gating значительно снижает риск плохих рекомендаций.

---

### 8. User interest modeling (2 из 15)

| Отчёт | Подход |
|-------|--------|
| 8 | MAP-инференс вектора интереса θ по истории сессии |
| 2 | Recency weight `0.8^i` для anchor tracks |

---

## Повторяющиеся идеи (dupicates count)

| Идея | Count | Отчёты |
|------|-------|--------|
| Гибрид SasRec+LightFM+HSTU | 4 | 3, 7, 13, 14 |
| Artist diversity penalty | 5 | 3, 4, 6, 11, 13 |
| Content эмбеддинги | 4 | 5, 6, 11, 15 |
| RRF | 2 | 7, 13 |
| Transition matrix | 2 | 12, 15 |

---

## Что было уникальным / редким

1. **Report 8** — MAP-session Bayesian inference (единственный fully online ML подход)
2. **Report 5** — TF-IDF без обучения (чисто контентный I2I)
3. **Report 12** — простейший transition prior без обучения эмбеддингов
4. **Report 2** — session-aware reranker без ML (чистые эвристики)

---

## Что могло быть упущено

1. **Нет использования sequential models** (SASREC, HSTU как candidate source, но не как reranker)
2. **Нет graph-based подходов** (хотя в курсе был GCN/GNN)
3. **Нет cross-session learning** — все учатся на bootstrap логах, нет continuous learning
4. **Мало кто экспериментировал с weight tuning** — большинство используют фиксированные веса w/(k+rank)
5. **Не исследовали cold-start** — что делать с новыми пользователями/треками

---

## Наиболее комплексные решения

1. **Report 6** (+27.57%) — LightFM retrieval + quantile LightGBM + MMR
2. **Report 14** (+24.55%) — LightGBM LambdaRank + 24 фичи + confidence gate
3. **Report 10** (+21.48%) — Hybrid I2I с LogReg blender + artist-diverse pick

## Наименее комплексные (но успешные)

1. **Report 5** (+35.10%) — TF-IDF content I2I (простейший подход!)
2. **Report 12** (+6.20%) — Transition prior (почти без ML)
3. **Report 2** (+16.46%) — Gated heuristic reranking
