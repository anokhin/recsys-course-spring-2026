# HW2. ML Hybrid Diversity Ranker for Botify

## Abstract

В тритменте реализован быстрый ML-гибридный ранжировщик для музыкального сервиса Botify. Контрольная группа оставлена без изменений и получает `SasRec-I2I`. Тритмент строит candidate pool из нескольких обученных моделей (`SasRec-I2I`, `LightFM-I2I`, `HSTU user recommendations`), а затем переупорядочивает кандидатов по implicit feedback пользователя: времени прослушивания последних треков, давности anchor-трека, согласию нескольких моделей и diversity-штрафу за повтор артиста. Основная гипотеза: в музыкальных рекомендациях чистый I2I часто уходит в цепочки одного артиста, а diversity-aware ML reranking должен увеличить `mean_time_per_session`.

## Детали реализации

Код находится в `botify/botify/recommenders/ml_hybrid.py`. На каждом запросе `/next/<user>` сервер сначала сохраняет последний playback в Redis, затем для тритмента читает последние события пользователя и строит скоринг кандидатов. SasRec, LightFM и HSTU используются как обученные candidate generators; итоговый выбор делает отдельный ранжировщик. Вес anchor-трека зависит от `time`, поэтому хорошо дослушанные треки сильнее влияют на следующий выбор, а плохие треки почти не тянут за собой похожие рекомендации.

Схема:

```text
Redis listen history
        |
        v
positive anchors + recent feedback
        |
        +--> SasRec-I2I candidates
        +--> LightFM-I2I candidates
        +--> HSTU user candidates
        |
        v
agreement + recency + listen-time scoring
        |
        v
artist diversity reranking
        |
        v
recommended track
```

