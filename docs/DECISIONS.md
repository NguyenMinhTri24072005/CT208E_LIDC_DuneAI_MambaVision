# Decisions

## 2026-09-16 — Use `SOURCES` as the capstone Git root

- **Decision:** Version the complete capstone from `SOURCES`; treat `Dataset_Preview` and `LIDC_MambaVision_Project` as modules.
- **Reason:** `Dataset_Preview` represents only the early data-exploration stage, not the complete research project.
- **Preservation:** The former Dataset Preview repository history was saved as a verified Git bundle in the sibling `GIT_BACKUPS` directory before its nested metadata was removed.

## 2026-09-16 — Vendor MambaVision source

- **Decision:** Include MambaVision as vendored source under `LIDC_MambaVision_Project/third_party/MambaVision`, not as a nested repository or submodule at this stage.
- **Reason:** The capstone currently integrates local MambaVision changes. Exact upstream provenance, commit and recovery artifacts are recorded in `docs/THIRD_PARTY.md`.

## 2026-09-16 — Keep research artifacts outside Git

- **Decision:** Exclude medical datasets, DICOM/NIfTI, downloaded annotations, processed data, patient-derived visualizations, checkpoints, weights, caches and experiment/runtime outputs.
- **Reason:** These files are sensitive, generated or too large for ordinary Git/GitHub storage. Source, configuration, documentation, tests and cleaned notebook source remain versioned.

## 2026-09-16 — Treat unverified architecture as planned work

- **Decision:** Distinguish design specifications and target file structures from implemented, tested functionality.
- **Reason:** The repository does not yet contain evidence for a complete end-to-end training/evaluation pipeline or validated capstone metrics.
