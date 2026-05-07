import pandas as pd
import json
import scipy.stats as stats
import zlib

def get_group(user_id):
    salt = "HSTU" # Твоя соль из experiment.py
    # Именно так Botify делит группы
    value = zlib.crc32(f"{user_id}{salt}".encode()) & 0xffffffff
    return 'T1' if value % 2 == 0 else 'C'

# Читаем данные
data = []
with open('data.json', 'r') as f:
    for line in f:
        data.append(json.loads(line))
df = pd.DataFrame(data)

# Группируем
df['group'] = df['user'].apply(get_group)
user_stats = df.groupby(['user', 'group'])['time'].sum().reset_index()

c = user_stats[user_stats['group'] == 'C']['time']
t = user_stats[user_stats['group'] == 'T1']['time']

mean_c, mean_t = c.mean(), t.mean()
lift = (mean_t - mean_c) / mean_c * 100
p_val = stats.ttest_ind(c, t).pvalue

print("\n=== РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА ===")
print(f"Control mean: {mean_c:.2f}")
print(f"Treatment mean: {mean_t:.2f}")
print(f"Relative Lift: {lift:+.2f}%")
print(f"P-value: {p_val:.6f}")

if p_val < 0.05 and lift > 0:
    print("\n✅ СТАТИСТИЧЕСКИ ЗНАЧИМО! Можно сдавать!")
else:
    print("\n❌ Нужно больше данных или более агрессивный рекоммендер.")