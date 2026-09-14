### Table 6. LUAD-CT-Survival external evaluation (binary long/short endpoint; diagnostic, not confirmatory).

| Model | AUC (95% CI) | AUC after sign reversal | Accuracy | F1 |
|---|---|---|---|---|
| Radiomics | 0.587 (0.415–0.775) | 0.412 | 0.500 | 0.091 |
| Embedding | 0.525 (0.346–0.699) | 0.475 | 0.500 | 0.167 |
| Joint | 0.568 (0.385–0.740) | 0.432 | 0.500 | 0.000 |

**Note.** Models were fitted on LUNG1 survival data using the radiomic and embedding blocks only (80 features, no clinical variables) and without batch correction. The classification threshold was the median hazard in the training cohort. n = 40 patients (20 long, 20 short). All confidence intervals include 0.5; this analysis is reported as diagnostic and not confirmatory.
