import pandas as pd
import json
import scipy.stats as stats

data = []
with open('data.json', 'r') as f:
    for line in f:
        data.append(json.loads(line))
df = pd.DataFrame(data)

# Если ты добавил поле в Datum, оно будет в колонке 'extra' или последней.
# Если нет — давай просто попробуем инвертировать u % 2, вдруг там T1 это нечетные
df['group'] = df['user'].apply(lambda u: 'T1' if u % 2 != 0 else 'C') 

user_stats = df.groupby(['user', 'group'])['time'].sum().reset_index()
c = user_stats[user_stats['group'] == 'C']['time']
t = user_stats[user_stats['group'] == 'T1']['time']

print(f"Control: {c.mean():.2f}, Treatment: {t.mean():.2f}")
print(f"Lift: {(t.mean()-c.mean())/c.mean()*100:.2f}%")
print(f"P-value: {stats.ttest_ind(c, t).pvalue:.6f}")