# Отчёт: Улучшение Botify через онлайн‑ML reranker (contextual bandit)

## i. Abstract: основную идею эксперимента (1 параграф)
В тритменте я заменяю ручное ранжирование на **онлайн‑ML модель**, которая учится по логам сервиса предсказывать ожидаемое время прослушивания трека в текущем контексте пользователя и выбирать трек с максимальным ожидаемым reward. В контроле показывается SasRec‑I2I (как в условии). Гипотеза: персонализированное ML‑ранжирование кандидатов (с небольшой exploration) статистически значимо увеличит \(mean\_session\_time\).

## ii. Детали: минимум того, что нужно знать, чтобы разобраться в реализации (1-2 параграфа + диаграмма)
Тритмент — это ML‑reranker поверх набора кандидатов. Кандидаты собираются без использования `sim/data/*`: (1) user‑based списки из `botify/data/hstu_recommendations.json`, (2) item‑to‑item кандидаты из `botify/data/lightfm_i2i.jsonl` по “якорям” из истории пользователя, (3) небольшая доля случайных треков для exploration/страховки. Для каждого (контекст, кандидат) строятся признаки из `botify/data/tracks.json`: агрегаты по последним трекам (топ‑жанры/артисты/муды, доля скипов, среднее время) + признаки кандидата (artist_id/genres/mood/year bucket/country/fans) + лёгкие cross‑фичи. Модель — онлайн‑линейная регрессия (SGD) с feature hashing в фиксированное пространство признаков; таргет \(y=\log(1+\text{listen\_time})\).

Онлайн‑обучение реализовано честно: сервис сохраняет в Redis “последний рекомендованный трек + признаки на момент рекомендации” и обновляет веса, когда этот трек возвращается на следующем запросе как `prev_track` с `prev_track_time`. Выбор — \(\epsilon\)-greedy: обычно берём argmax по предсказанию, иногда — случайного кандидата для exploration.

Диаграмма:

`history (redis)` → `candidates (HSTU + LightFM-i2i + explore)` → `features (ctx+item, hashing)` → `score` → `choose` → `log` → `SGD update on next request`

## iii. Результаты A/B эксперимента - в табличке как на семинарах (1 параграф + табличка)
Эксперимент проводился честно: **Control = SasRec‑I2I** (`Treatment.C`), **Treatment = Online‑ML reranker** (`Treatment.T1`) в эксперименте `HSTU`. Метрика \(mean\_session\_time\) считалась по `botify/log/data.json` как сумма поля `time` внутри сессии (сессия заканчивается событием `last`), затем среднее по сессиям в каждой группе. Прогон: `python -m sim.run --episodes 200 --config config/env.yml single --recommender remote --seed 3133`.

| treatment | metric | effect_pct | lower_pct | upper_pct | control_mean | treatment_mean | significant |
|-----------|--------|-----------|----------|----------|-------------|---------------|------------|
| T1 | mean_time_per_session | 206.56 | 168.23 | 244.89 | 1.4321 | 4.3903 | True |

