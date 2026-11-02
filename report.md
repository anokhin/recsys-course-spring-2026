## Homework 2 Report

### Abstract

Честный A/B: контроль — SasRec-I2I (соседи `recommend_to_items` + история в Redis и сэмпл контекста ∝ listen time), тритмент — тот же I2I-пайплайн с кандидатами SasRec, но следующий трек выбирается softmax-реранком ContextualRanker (`context · track`, MSE по доле прослушки). Ранкер обучен на логах симулятора (без `sim/data`), в сервис идут только `sasrec_i2i.jsonl` и эмбеддинги `.npy`. Цель — поднять `mean_time_per_session`.

### Details

Данные собраны прогоном StickyArtist → interactions `(user, track, time)`. SasRec (rectools, softmax) экспортирован в I2I top-K. ContextualRanker (dim=96) дообучен на control-arm логах; онлайн скорим кандидатов SasRec, температура 0.08, смесь 85% softmax + 15% uniform, контекст из последних 20 треков с весом `time^1.5`. Эксперимент один: `HW2` HALF_HALF (`C` / `T1`).

```text
sim -> nginx:5001 -> botify /next
                      |-- HW2.C  -> I2I(SasRec neighbors, shuffle unseen)
                      `-- HW2.T1 -> RankedI2I(SasRec candidates,
                                              ContextualRanker softmax)
logs -> analyze_ab.py -> mean_time_per_session
```

### A/B Results

Локально `SEED=31312`, `EPISODES=30000`, два независимых прогона. По `mean_time_per_session` тритмент значимо лучше контроля; |Δ effect| между прогонами 2.72 п.п. (≤ 10%).

| treatment | metric | control_mean | treatment_mean | effect_pct | lower_pct | upper_pct | significant |
|-----------|--------|--------------|----------------|------------|-----------|-----------|-------------|
| T1 | mean_time_per_session | 4.6170 | 5.6005 | 21.30 | 18.23 | 24.38 | True |
| T1 | mean_time_per_session (run2) | 4.4783 | 5.3105 | 18.58 | 15.80 | 21.36 | True |
