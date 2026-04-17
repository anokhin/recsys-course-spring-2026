# Homework 2 Report

## Abstract

В этом решении treatment построен как **session-aware linear meta-ranker** поверх объединённого множества кандидатов из трёх источников: `SasRec-I2I`, `LightFM-I2I` и `HSTU`. В отличие от базового `SasRec-I2I`, который фактически выбирает рекомендацию от одного якорного трека, новый рекомендер агрегирует несколько последних прослушиваний пользователя и учитывает структуру текущей сессии. Цель модели — повысить `mean_session_time` за счёт более устойчивого выбора следующего трека: усиливать кандидатов, которые поддерживаются несколькими источниками, и снижать score для кандидатов, ведущих к artist fatigue.

## Details

### Pipeline

```text
listen history
    |
    v
recent anchors (up to 3 unique tracks)
    |
    +--> SasRec-I2I candidates
    +--> LightFM-I2I candidates
    +--> HSTU user candidates
             |
             v
     union candidate set
             |
             v
feature extraction
(in_source, reciprocal ranks, source votes,
same_artist_as_last, artist_seen_count,
fresh_artist, anchor_weight_sum, ...)
             |
             v
linear score = b + Σ w_i x_i
             |
             v
top-1 deterministic recommendation
```

Treatment работает полностью детерминированно. Для каждого кандидата считаются признаки по текущей сессии: присутствие в источниках, лучшие reciprocal-rank значения, поддержка от последнего и предпоследнего трека, число уже встреченных в сессии артистов, а также признаки свежести артиста. Затем используется линейный скорер с коэффициентами из `linear_ranker_weights.json`. Контроль в эксперименте оставлен без изменений: `SasRec-I2I`.

Основные файлы реализации:
- `botify/botify/recommenders/linear_session_ranker.py`
- `botify/botify/server.py`
- `botify/data/linear_ranker_weights.json`

Опционально для дальнейшего улучшения качества добавлен `script/train_linear_ranker.py`: он позволяет переоценить коэффициенты модели по собранным логам предыдущих прогонов без изменения serving-кода.

## A/B Results

Ниже нужно вставить результат из комментария GitHub Actions после прогона PR.

| metric | control | treatment | effect | p-value |
|---|---:|---:|---:|---:|
| mean_time_per_session | TODO | TODO | TODO | TODO |

Краткий вывод после CI: **TODO**.
