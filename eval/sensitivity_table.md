|                                      |   ('ret', 'mean') |   ('ret', 'std') |   ('recon_deflection_rate', 'mean') |   ('recon_deflection_rate', 'std') |   ('legit_fp_rate', 'mean') |   ('legit_fp_rate', 'std') |
|:-------------------------------------|------------------:|-----------------:|------------------------------------:|-----------------------------------:|----------------------------:|---------------------------:|
| ('detect_p_x0.5', 'no_deception')    |          -139.974 |            1.592 |                               0     |                              0     |                       0     |                      0     |
| ('detect_p_x0.5', 'qrdqn')           |            22.229 |           22.849 |                               0.983 |                              0.012 |                       0.006 |                      0.001 |
| ('detect_p_x0.5', 'static_rule')     |           -18.132 |            1.972 |                               0.727 |                              0.025 |                       0     |                      0     |
| ('detect_p_x1.5', 'no_deception')    |          -139.974 |            1.592 |                               0     |                              0     |                       0     |                      0     |
| ('detect_p_x1.5', 'qrdqn')           |            -0.434 |            1.046 |                               0.965 |                              0.038 |                       0.004 |                      0.003 |
| ('detect_p_x1.5', 'static_rule')     |           -39.51  |            1.97  |                               0.765 |                              0.025 |                       0     |                      0     |
| ('legit_rate_0.5', 'no_deception')   |          -116.041 |            2.336 |                               0     |                              0     |                       0     |                      0     |
| ('legit_rate_0.5', 'qrdqn')          |             3.487 |            3.268 |                               0.915 |                              0.008 |                       0     |                      0     |
| ('legit_rate_0.5', 'static_rule')    |           -33.103 |            1.334 |                               0.746 |                              0.016 |                       0     |                      0     |
| ('penalties_x1.5', 'no_deception')   |          -215.217 |            2.399 |                               0     |                              0     |                       0     |                      0     |
| ('penalties_x1.5', 'qrdqn')          |           -27.572 |           12.436 |                               0.95  |                              0.037 |                       0.002 |                      0.003 |
| ('penalties_x1.5', 'static_rule')    |           -71.066 |            2.957 |                               0.749 |                              0.022 |                       0     |                      0     |
| ('rewards_pos_x0.5', 'no_deception') |          -145.23  |            1.603 |                               0     |                              0     |                       0     |                      0     |
| ('rewards_pos_x0.5', 'qrdqn')        |           -38.022 |           11.8   |                               0.983 |                              0.012 |                       0.005 |                      0.001 |
| ('rewards_pos_x0.5', 'static_rule')  |           -54.212 |            1.77  |                               0.749 |                              0.022 |                       0     |                      0     |

QR-DQN return > static_rule (seeds won / total):

| condition        |   sum |   count |
|:-----------------|------:|--------:|
| detect_p_x0.5    |     3 |       3 |
| detect_p_x1.5    |     3 |       3 |
| legit_rate_0.5   |     3 |       3 |
| penalties_x1.5   |     3 |       3 |
| rewards_pos_x0.5 |     3 |       3 |