# Отчёт по домашнему заданию 2

## Abstract

Тритмент — двухстадийный i2i: **ALS-retrieval (`implicit.als`)** на listen-time-взвешенной
матрице user×item даёт пул из 500 кандидатов на каждый anchor; **Cox-PH survival-реранкер**
(`lifelines.CoxPHFitter`) предсказывает риск скипа и пере-сортирует кандидатов.
Каждое наблюдение `(anchor → candidate, listen_time)` трактуется как survival-точка:
`event=1` если `listen_time<0.5` (трек скипнут — «смерть»), `event=0` иначе
(трек дослушан — right-censored). Финальный скор — `−log(partial_hazard)`:
ниже hazard ⇔ ниже шанс скипа ⇔ выше место в выдаче. Идея — заменить классический
LambdaRank/quantile-loss на цензурированную survival-регрессию,
которая корректно учитывает «дослушал до конца ≠ всё, что мог бы».

## Детали реализации

**Retrieval (`script/train_retrieval.py`).** ALS из `implicit` на user×item матрице
с конфиденсами `1 + 40·log1p(listen_time)`. Параметры — 64 фактора, 25 итераций,
λ=0.05. На выходе: top-15000 «горячих» anchor-ов и для каждого top-500 кандидатов
по dot-product item-факторов. Параллельно сохраняем `item_embs.npz`
для использования как фичи в реранкере.

**Reranker (`script/train_survival.py`).** Шесть pair-level фичей с z-нормализацией
перед фитом:

| Feature | Описание |
|---|---|
| `pmi(a,c)` | Pointwise mutual information переходов c laplace-сглаживанием |
| `beta_binom_completion(a,c)` | Posterior mean Beta-Binomial completion-rate с empirical-Bayes prior (method-of-moments из глобального распределения) |
| `dot_score(a,c)` | Dot-product ALS item-факторов |
| `same_artist(a,c)` | Boolean — артист совпадает |
| `pop_score(c)` | `log(1 + cand_plays)` — популярность кандидата |
| `cand_skip_rate(c)` | Smoothed skip-rate кандидата `(skips+1)/(plays+2)` |

`CoxPHFitter(penalizer=0.01)`, ridge-регуляризация для устойчивости при коррелированных
PMI/pop_score. Train-set отсортирован по `(duration, event)` для детерминизма
Efron tie-breaking. Финальное ранжирование — top-200 по `−log(partial_hazard)`,
опционально с Gumbel-Top-k sampling на ранках ≥ 20 (`τ=0` в проде).

## A/B эксперимент

Эксперимент `LEARNED_SURVIVAL`, разбивка `HALF_HALF`, `EPISODES=30000`, `seed=31312`.
Контроль — SasRec-I2I, тритмент — ALS retrieval + Cox PH reranker, fallback на SasRec-I2I.

| Метрика | Контроль | Тритмент | Эффект | 95% CI | Значимо |
|---|---|---|---|---|---|
| **mean_time_per_session** | 7.00 с | **8.17 с** | **+16.69%** | [+14.33%, +19.06%] | да |
| mean_tracks_per_session | 11.97 | 13.14 | +9.83% | [+8.36%, +11.30%] | да |
| time | 21.68 с | 25.08 с | +15.67% | [+12.46%, +18.89%] | да |
| sessions | 3.16 | 3.13 | −0.83% | [−2.89%, +1.23%] | нет |
| mean_request_latency | 0.55 мс | 0.60 мс | +9.08% | [−6.57%, +24.74%] | нет |

Главная метрика выросла на **+16.69%** при значимости. Воспроизводимо: повторный
прогон (`make clean && make setup && make run`) дал эффект `+17.10%`,
delta=0.41% ≪ 10% порог. На 200k обучающих парах все 6 фичей значимы;
самые сильные — `beta_binom_completion` (`coef ≈ −1.48`, `p ≪ 10⁻¹⁵`)
и `same_artist` (`coef ≈ −0.11`, `p ≈ 10⁻¹⁶⁵`). Положительный знак у `pop_score`
и `pmi` — после контроля по completion-rate популярные/«ожидаемые» переходы
имеют чуть больший hazard скипа (эффект эксплуатационной усталости).
