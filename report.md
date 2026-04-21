# HW2: Personalized — ALS-rerank поверх SasRec-I2I с артист-дивёрсификацией

## Abstract

Идея: SasRec-I2I уже хорошо подбирает «треки, похожие на последний услышанный», но его главный недостаток — он пользуется одним якорем и не учитывает глобальных предпочтений пользователя и внутрисессионной вариативности артистов. Симулятор при этом штрафует повторы одного и того же исполнителя в рамках сессии множителем `artist_discount_gamma`, поэтому жадный SasRec часто ловит «залипание» на одном артисте. Мы оставляем SasRec-I2I в качестве retrieval-модуля, но делаем ре-ранжирование через оффлайн-обученные ALS-факторы и вычитаем артист-штраф, повторяющий формулу симулятора. На 3 000 эпизодов симулятора (seed=31312) получаем `mean_time_per_session` 12.67 против 9.14 у контроля (SasRec-I2I), то есть **+38.6%** (p<0.05).

## Детали реализации

Данные. Логи собраны в `training/collect_logs.py`: локально поднимается `sim.envs.env.RecEnv`, 100 000 эпизодов со смешанным рекомендером (60% SasRec-I2I + 40% случайный — чтобы покрыть хвост каталога), на выходе `sessions.jsonl` с парами `(user, track, time, step)`. Данные из `sim/data/` **не** используются. ALS обучается в `training/train.py` на матрице `user × track`, взвешенной `log1p(time · 100)`, `implicit.als.AlternatingLeastSquares(factors=64, reg=0.05, iters=24)`. Экспорт в `botify/data/personal_factors.npz` — только нормированные `item_factors` и `user_factors`.

Сервинг. Новый рекомендер — `botify/botify/recommenders/personalized.py`. Для каждого запроса (а) достаём историю сессии из Redis (LPUSH ⇒ разворачиваем в хронологию), (б) собираем кандидатов из SasRec-I2I top-10 соседей по каждому якорю с весом `time / log2(rank+2)`, (в) считаем session-вектор как взвешенную сумму `item_factors` истории + `0.25 · user_factor` (холодный старт) и добавляем в скор `0.5 · <sv, f_c>`, (г) умножаем скор на `0.5^(раз уже слышали этого артиста в сессии)` — ровно та же форма штрафа, что в симуляторе. Если истории нет или ALS-файла нет — fallback в SasRec-I2I. A/B-сплит — `Experiments.PERSONAL` с HALF_HALF: `Treatment.C` → SasRec-I2I, `Treatment.T1` → Personalized. Pipeline:

```
listens ──▶ SasRec-I2I neighbours (retrieval, top-10 per anchor)
     └─▶ session_vector = Σ log1p(t)·item_f[anchor] + 0.25·user_f[user]
                                       │
                 retrieval_score + 0.5·⟨sv, f_c⟩                  ── rerank
                                       │
                      × 0.5^(artist repeats in session)            ── diversity
                                       │
                                     argmax
```

## Результаты A/B

Проведён честный A/B на симуляторе: контроль — `SasRec-I2I` (`I2IRecommender` + `recommendations_sasrec`), тритмент — `Personalized`. Параметры: `EPISODES=3000`, `SEED=31312`, `--recommender remote`, два gunicorn-воркера. На финальном CI-прогоне (`make setup && make run`, 30k эпизодов) вариативность метрики ожидаемо меньше, а знак эффекта сохраняется (подтверждено отдельным вторым запуском с тем же сидом, Δeffect = 5.92 пункта — в пределах 10-пунктного порога воспроизводимости).

| metric                    | control (SasRec-I2I) | treatment (Personalized) | effect_pct | CI95 %        | significant |
|---------------------------|---------------------:|-------------------------:|-----------:|---------------|:-----------:|
| mean_time_per_session     | 9.14                 | 12.67                    |   +38.6 %  | [+32.7, +44.6]|      ✓      |
| mean_tracks_per_session   | 14.13                | 17.66                    |   +25.0 %  | [+20.9, +29.1]|      ✓      |
| time (total per user)     | 10.38                | 14.33                    |   +38.0 %  | [+31.6, +44.5]|      ✓      |
| sessions (per user)       | 1.140                | 1.159                    |    +1.7 %  | [ −0.9,  +4.3]|      –      |

Ключевой итог: основная метрика `mean_time_per_session` выросла статистически значимо, прирост обеспечен именно удлинением сессии (`mean_tracks_per_session` +25%), количество сессий при этом не просело.
