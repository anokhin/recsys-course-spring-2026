## Homework 2 Report

### Abstract

Контроль — SasRec-I2I: соседи из `recommend_to_items`, контекст сэмплируется из истории в Redis пропорционально listen time. Тритмент — те же кандидаты SasRec, но выбор следующего трека делает ContextualRanker (dot-product эмбеддингов, MSE по доле прослушки, softmax). Модели обучены на логах симулятора; в сервис кладутся `sasrec_i2i.jsonl` и эмбеддинги `.npy`. Оптимизируем `mean_time_per_session`.

### Details

Логи собраны с StickyArtist. SasRec (rectools, softmax) экспортирован в I2I top-K. ContextualRanker (dim=96) обучен на парах (context, track) → time; в онлайне скорим кандидатов SasRec, T=0.08, смесь 0.85 softmax + 0.15 uniform, контекст из последних 20 треков с весом `time^1.5`. Сплит: эксперимент `HW2`, HALF_HALF, C vs T1.

```text
sim -> nginx:5001 -> botify /next
                      |-- C  -> I2I(SasRec, shuffle)
                      `-- T1 -> RankedI2I(SasRec + ContextualRanker)
logs -> analyze_ab.py -> mean_time_per_session
```

### A/B Results

`SEED=31312`, `EPISODES=30000`, два прогона. По `mean_time_per_session` тритмент значимо лучше; расхождение эффектов 2.42 п.п.

| treatment | metric | control_mean | treatment_mean | effect_pct | lower_pct | upper_pct | significant |
|-----------|--------|--------------|----------------|------------|-----------|-----------|-------------|
| T1 | mean_time_per_session | 4.6408 | 5.5945 | 20.55 | 17.52 | 23.59 | True |
| T1 | mean_time_per_session (run2) | 4.4821 | 5.2948 | 18.13 | 15.38 | 20.89 | True |
