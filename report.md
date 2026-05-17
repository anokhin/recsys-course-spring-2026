# Домашнее задание 2: Online ML Ranker

## Основная идея

Я реализовал для Botify online ML-реранкер, который улучшает исходный `SasRec-I2I` за счет персонализации по короткой истории пользователя. В контрольной группе используется неизмененный `SasRec-I2I`. В treatment сначала собирается небольшой набор кандидатов из нескольких item-to-item моделей, а затем обученная логистическая модель выбирает лучший следующий трек по признакам истории, рангов кандидатов и метаданных треков. Это не ручная эвристика: итоговое решение принимает ML-модель, обученная на логах взаимодействий.

## Детали

Модель обучалась offline на логах взаимодействий Botify. Для каждого состояния пользователя строились кандидаты из `SasRec-I2I`, `LightFM-I2I` и моего DLRM-rerank I2I, после чего кандидат размечался как хороший, если следующий прослушанный трек получил достаточно большое время прослушивания. Обученная модель сохраняется в `botify/data/online_ranker_model.json` как набор коэффициентов, поэтому во время пользовательского запроса сервис не зависит от версии `sklearn` и не загружает тяжелый ML-фреймворк.

В online-сервисе `botify/botify/recommenders/online_ranker.py` читает последние прослушивания пользователя из Redis, собирает кандидатов из трех I2I-источников и считает признаки: длина истории, среднее и последнее время прослушивания, совпадение артиста/жанров/mood, ранги кандидата в SasRec/LightFM/DLRM, а также глобальную статистику трека по логам. Если confidence модели низкий, используется fallback на стабильный DLRM-rerank. В A/B-тесте `HOMEWORK2` контроль `C` использует `sasrec_i2i_recommender`, а treatment `T1` использует `online_ranker_recommender`.

```text
                    OFFLINE TRAINING

  Логи взаимодействий Botify
          |
          v
  Обучение logistic ranker
          |
          v
  online_ranker_model.json


                    ONLINE SERVING

  Последние треки пользователя из Redis
          |
          v
  Candidate pool:
    - SasRec-I2I
    - LightFM-I2I
    - DLRM-rerank I2I
          |
          v
  Feature extraction
          |
          v
  Online ML Ranker
          |
          v
  Следующий трек


                    A/B TEST

        Control C                          Treatment T1
   sasrec_i2i_recommender              online_ranker_recommender
```

## Результаты A/B эксперимента

A/B-тест был проведен в эксперименте `HOMEWORK2` на CI-like запуске с `30000` эпизодов и `seed=31312`. Целевая метрика статистически значимо улучшилась: `mean_session_time` выросла на `31.60%` при `pvalue = 1.03e-307`.

| metric | control_mean | treatment_mean | effect_% | pvalue | significant |
|---|---:|---:|---:|---:|---|
| mean_session_time | 6.80656 | 8.95711 | 31.5953 | 1.02887e-307 | True |
| mean_tracks_per_session | 11.7470 | 13.8808 | 18.1643 | 2.90642e-247 | True |
| mean_request_latency | 0.000399507 | 0.000798031 | 99.7540 | 0 | True |
| sessions | 15262 | 15090 | -1.12698 | nan | False |
