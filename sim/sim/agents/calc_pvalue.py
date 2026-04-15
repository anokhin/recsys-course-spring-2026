import numpy as np
from scipy import stats

control_mean = 2.70329
control_sem = 0.0212004
treatment_mean = 2.82922
treatment_sem = 0.0218327

n_control = 5008
n_treatment = 4916

control_std = control_sem * np.sqrt(n_control)
treatment_std = treatment_sem * np.sqrt(n_treatment)

np.random.seed(42)
control = np.random.normal(control_mean, control_std, n_control)
treatment = np.random.normal(treatment_mean, treatment_std, n_treatment)

t_stat, p_value = stats.ttest_ind(treatment, control)

print(f"Control:  {control_mean:.4f} ± {control_sem:.4f}")
print(f"Treatment: {treatment_mean:.4f} ± {treatment_sem:.4f}")
print(f"Прирост: {((treatment_mean/control_mean - 1) * 100):.2f}%")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.6f}")
print(f"Значимо: {'ДА (p < 0.05)' if p_value < 0.05 else 'НЕТ'}")