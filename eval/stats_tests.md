| comparison | metric | test | statistic | p-value |
|---|---|---|---|---|
| qrdqn vs no_deception | ret | paired t | 65.712 | 0.0000 |
| qrdqn vs no_deception | recon_deflection_rate | paired t | 284.296 | 0.0000 |
| qrdqn vs no_deception | legit_fp_rate | paired t | 4.426 | 0.0115 |
| qrdqn vs random | ret | paired t | 106.512 | 0.0000 |
| qrdqn vs random | recon_deflection_rate | paired t | 35.836 | 0.0000 |
| qrdqn vs random | legit_fp_rate | paired t | -121.587 | 0.0000 |
| qrdqn vs static_rule | ret | paired t | 25.567 | 0.0000 |
| qrdqn vs static_rule | recon_deflection_rate | paired t | 16.253 | 0.0001 |
| qrdqn vs static_rule | legit_fp_rate | paired t | 4.426 | 0.0115 |
| qrdqn vs dqn | ret | paired t | 1.848 | 0.1384 |
| qrdqn vs dqn | recon_deflection_rate | paired t | -2.027 | 0.1126 |
| qrdqn vs dqn | legit_fp_rate | paired t | -4.830 | 0.0085 |

n = 5 seeds (paired by seed). With n=5, the smallest possible two-sided Wilcoxon p-value is 0.0625, so the paired t-test is used when differences pass Shapiro-Wilk.