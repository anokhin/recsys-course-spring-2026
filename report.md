## Homework 2 Report

### Abstract

Учился только на логах `botify`. В качестве модели использую reward-weighted transition model первого порядка: по восстановленным успешным рекомендациям строю разреженную матрицу переходов `M[a, b] = sum(time)`, где `a = prev_track`, `b = recommended_track`, а `time` это время прослушивания следующего трека. В онлайне для текущего `prev_track = a` выбираю трек с максимальным весом `M[a, b]` среди еще не прослушанных пользователем; если для состояния данных нет, использую глобальный prior `g[b] = sum(time)` по всем успешным переходам.

### Details

Брал логи `next/last` из `botify`, собранные в отдельных прогонах симулятора. Из них восстановил 305,214 положительных примеров вида `(user, prev_track, recommended_track, reward_time)`, где рекомендация считается успешной, если следующим реально прослушанным треком оказался именно рекомендованный.

```text
botify logs
  -> восстанавливаю успешные рекомендации
  -> получаю строки: (user, prev_track, recommended_track, reward_time)
  -> суммирую reward по (prev_track, track) и отдельно по track
  -> сохраняю артефакт в JSONL для Redis
  -> в рантайме:
       беру ranking для track:{prev_track}
       -> убираю уже прослушанные треки
       -> если данных нет, беру global ranking
```

### Results

Финальный A/B-прогон выполнен на 10,000 эпизодах: `C = SasRec-I2I`, `T1 = my-model`. По основной метрике `time` получился статистически значимый прирост.

| treatment | metric | effect_pct | upper_pct | lower_pct | control_mean | treatment_mean | significant |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| T1 | time | 6.51 | 10.37 | 2.64 | 11.5266 | 12.2766 | True |
| T1 | sessions | -1.08 | 1.43 | -3.58 | 1.5852 | 1.5681 | False |
| T1 | mean_request_latency | -1.80 | -1.44 | -2.15 | 0.5492 | 0.5393 | True |
| T1 | mean_tracks_per_session | 3.84 | 5.68 | 1.99 | 12.3233 | 12.7959 | True |
| T1 | mean_time_per_session | 6.20 | 9.06 | 3.34 | 7.3512 | 7.8072 | True |
