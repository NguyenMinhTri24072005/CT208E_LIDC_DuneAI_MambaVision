# Project Status

Updated: 2026-09-16

## Current Phase

Early-stage LIDC-IDRI exploration and MambaVision integration research: TCIA/NBIA data access, Study/Series inspection, DICOM reading, profiling, CT visualization, annotation exploration, and model-environment sanity work.

## Completed

- Dataset Preview implementations exist for TCIA API access, selective download/extraction, DICOM sorting, Hounsfield-unit conversion, annotation parsing, and visualization.
- Dataset profiling and canonical CT Quick Viewer notebook sources are present with stored execution outputs removed.
- MambaVision source is vendored under `LIDC_MambaVision_Project/third_party/MambaVision` with upstream provenance and local modifications documented.
- The former `Dataset_Preview` repository history is preserved in a verified Git bundle outside `SOURCES` before root migration.
- Repository-wide data, secret, cache, checkpoint, and output exclusions are defined at the `SOURCES` root.

## In Progress

- Validating the MambaVision sanity/EDA workflow and dependency compatibility.
- Defining the implemented capstone pipeline beyond the current package scaffolding.
- Connecting verified LIDC-IDRI preprocessing outputs to model inputs.
- Cần xác nhận: provenance and reproducibility of the local checkpoint under `LIDC_MambaVision_Project/checkpoints/`.

## Next Actions

- Run Dataset Preview tests and the MambaVision sanity notebook in documented clean environments.
- Validate the local MambaVision compatibility changes and pure-PyTorch selective-scan fallback.
- Implement and test the project-specific dataset, model adapter, training, evaluation, and inference modules described by the design documents.
- Define reproducible dataset splits, experiment IDs, metrics, and external artifact storage.
- Record every training/evaluation run in `docs/EXPERIMENTS.md`.

## Blockers

- Cần xác nhận: final task definition, dataset split, compute environment, and acceptance metrics.
- Cần xác nhận: exact origin, configuration, metrics, and reproducibility of the local checkpoint.

## Latest Results

No training or evaluation metric has been verified during repository migration. A local checkpoint exists but is excluded from Git and must not be treated as a validated result until its provenance and reproduction procedure are documented.

## Important Files

- `README.md` — repository-wide scope and current evidence.
- `MambaVision_Medical_AI_Research_Specification.md` — proposed research/system design.
- `Dataset_Preview/README.md` — data-exploration module documentation.
- `Dataset_Preview/src/lidc/` — current LIDC data-access and preview implementation.
- `LIDC_MambaVision_Project/README.md` — target architecture notes for the MambaVision workspace.
- `LIDC_MambaVision_Project/notebooks/01_mambavision_sanity_check.ipynb` — model-environment sanity/EDA notebook source.
- `docs/THIRD_PARTY.md` — vendored MambaVision provenance.

## Current Branch

`main`
