### Table 2. Model performance: internal cross-validation and external validation.

| Model | Internal CV | External C-index (95% CI) | AUC 12m | AUC 24m | AUC 36m |
|---|---|---|---|---|---|
| Clinical | 0.536 ± 0.028 | 0.593 (0.522–0.666) | 0.667 | 0.672 | 0.592 |
| Radiomics | 0.575 ± 0.035 | 0.427 | 0.443 | 0.350 | 0.420 |
| Embedding | 0.550 ± 0.014 | 0.581 (0.504–0.650) | 0.551 | 0.588 | 0.603 |
| Joint | 0.582 (0.565–0.598) | 0.507 (0.431–0.580) | 0.522 | 0.494 | 0.501 |

**Note.** Internal CV for the joint model reports the 50-seed x 5-fold bootstrap mean with 95% CI; other internal entries report single-run 5-fold mean ± SD. External C-index is bootstrap mean with 95% CI. AUC values are point estimates at 12, 24 and 36 months. Point estimates for external C-index (0.594 / 0.427 / 0.584 / 0.509) are within sampling noise of the bootstrap means (delta <= 0.003).
