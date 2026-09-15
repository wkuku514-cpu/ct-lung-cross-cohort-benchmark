### Table 4. Decomposition of cross-cohort performance loss.

| Condition | Internal CV C-index | External C-index | Delta |
|---|---|---|---|
| ComBat | 0.575 | 0.509 | -0.066 |
| No ComBat | 0.575 | 0.458 | -0.117 |

**Note.** Internal CV is single-run 5-fold in LUNG1; external C-index is the joint model in the primary external cohort. In-cohort cross-validation is identical before and after ComBat because per-fold standardization removes the per-batch affine correction within the source cohort.
