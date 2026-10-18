## Macro-F1 over 5 seeds (5-fold stratified CV each)

| model   |   mean |    std |
|:--------|-------:|-------:|
| RF      | 0.9214 | 0.0073 |
| XGB     | 0.9122 | 0.0078 |

### RF (seed 42)
```
              precision    recall  f1-score   support

         apt      0.923     0.905     0.914       400
  downgrader      0.902     0.897     0.900       400
     flooder      0.907     0.907     0.907       400
       legit      0.897     0.980     0.937       400
     scanner      0.866     0.887     0.877       400
    stealthy      0.978     0.887     0.931       400

    accuracy                          0.911      2400
   macro avg      0.912     0.911     0.911      2400
weighted avg      0.912     0.911     0.911      2400
```

### XGB (seed 42)
```
              precision    recall  f1-score   support

         apt      0.907     0.905     0.906       400
  downgrader      0.907     0.905     0.906       400
     flooder      0.902     0.900     0.901       400
       legit      0.901     0.935     0.918       400
     scanner      0.868     0.875     0.872       400
    stealthy      0.932     0.897     0.915       400

    accuracy                          0.903      2400
   macro avg      0.903     0.903     0.903      2400
weighted avg      0.903     0.903     0.903      2400
```

## ATT&CK mapping

| class      | MITRE ATT&CK                                                  |
|:-----------|:--------------------------------------------------------------|
| scanner    | TA0043 Reconnaissance / T1595 Active Scanning                 |
| downgrader | TA0005 Defense Evasion / T1562.010 Downgrade Attack           |
| flooder    | TA0040 Impact / T1499 Endpoint Denial of Service              |
| apt        | Multi-stage: TA0043 -> TA0005 -> TA0040                       |
| stealthy   | TA0043 Reconnaissance / T1595 via protocol-conformant probing |
| legit      | benign                                                        |

Note: sessions come from the simulated attacker profiles, so these scores validate the pipeline on the modelled behaviours, not detection of real-world attackers.