# Data sources

This study uses five publicly available collections distributed by The Cancer Imaging Archive (TCIA). **No imaging data are redistributed in this repository.** Download each collection from TCIA under its own data use agreement before running the pipeline.

| Collection | Role in this study | n | TCIA DOI | Access | Licence | Download date |
|---|---|---|---|---|---|---|
| NSCLC-Radiomics (LUNG1) | Development cohort | 422 | **[to be confirmed]** | TCIA collection page | **[to be confirmed]** | **[to be confirmed]** |
| NSCLC-Radiogenomics | Primary external cohort | 211 | **[to be confirmed]** | TCIA collection page | **[to be confirmed]** | **[to be confirmed]** |
| LungCT-Diagnosis | Imaging provenance audit | 61 | **[to be confirmed]** | TCIA collection page | **[to be confirmed]** | **[to be confirmed]** |
| QIN-LUNG-CT | Imaging provenance audit | 47 | **[to be confirmed]** | TCIA collection page | **[to be confirmed]** | **[to be confirmed]** |
| LUAD-CT-Survival | Second (diagnostic) evaluation | 40 | **[to be confirmed]** | TCIA collection page | **[to be confirmed]** | **[to be confirmed]** |

## Please complete before publishing this file

During preparation of the manuscript, the TCIA website and wiki were unreachable from the authoring environment, so the collection DOIs, licence types and download dates could not be verified online. Fill in each `[to be confirmed]` cell from the corresponding TCIA collection page, and cite each collection in the manuscript as required by TCIA. Do not insert DOIs from memory.

## Expected directory layout

The scripts assume the collections are extracted under a single working directory, one folder per collection, with a manifest CSV listing `dataset`, `patient_id`, `study_uid`, `series_uid`, `n_files` and `folder`. Adjust the path constants at the top of each script accordingly.

## Derived data

`tables/` contains aggregate summary statistics only (model performance, calibration, stability and intervention results). Individual-level feature tables are intentionally not distributed.
