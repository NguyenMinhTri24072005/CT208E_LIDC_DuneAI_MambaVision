# Capstone: MambaVision for LIDC-IDRI Lung CT Research

Dự án niên luận nghiên cứu khả năng ứng dụng MambaVision cho ảnh CT phổi, với LIDC-IDRI là dataset mục tiêu và TCIA/NBIA là nguồn truy cập dữ liệu. Repository hiện ở giai đoạn đầu: khám phá dữ liệu, kiểm tra Study/Series, đọc DICOM, profiling, visualization, nghiên cứu annotation và kiểm tra khả năng tích hợp backbone MambaVision.

> **Trạng thái:** Repository chưa chứng minh một training pipeline end-to-end, baseline hoàn chỉnh, kết quả đánh giá hoặc metric nghiên cứu đã được tái lập. Các phần này vẫn đang phát triển hoặc cần xác nhận.

## Phạm vi hiện có

- `Dataset_Preview/`: module khám phá LIDC-IDRI, gồm TCIA/NBIA API helpers, tải chọn lọc, đọc/sắp xếp DICOM, chuyển đổi HU, annotation parsing, visualization, notebook profiling và tests.
- `LIDC_MambaVision_Project/`: khu vực phát triển liên quan MambaVision. Hiện có dependency manifest, package scaffolding, một notebook sanity/EDA và source MambaVision được vendored.
- `LIDC_MambaVision_Project/third_party/MambaVision/`: snapshot source từ `NVlabs/MambaVision`, có local modifications được ghi nhận trong `docs/THIRD_PARTY.md`.
- `MambaVision_Medical_AI_Research_Specification.md`: đặc tả và roadmap nghiên cứu; đây là thiết kế mục tiêu, không phải bằng chứng rằng mọi thành phần đã được triển khai.
- `docs/`: trạng thái, quyết định, mẫu ghi experiment và provenance third-party cho toàn bộ capstone.

## Cấu trúc repository

```text
SOURCES/
├── Dataset_Preview/                  # Data access, profiling, DICOM/annotation exploration
├── LIDC_MambaVision_Project/         # MambaVision integration workspace
│   ├── notebooks/                    # Sanity/EDA notebook source
│   ├── src/                          # Current project package scaffolding
│   ├── third_party/MambaVision/      # Vendored upstream source with documented local changes
│   ├── requirements.txt              # Current dependency declarations
│   └── checkpoints/                  # Local model artifacts; excluded from Git
├── docs/                             # Project-wide status, decisions, experiments, provenance
└── MambaVision_Medical_AI_Research_Specification.md
```

## Thiết lập và kiểm tra

Dependency declarations for the MambaVision workspace are in `LIDC_MambaVision_Project/requirements.txt`. Compatibility and reproducibility across Windows, Linux and Colab still require validation.

The Dataset Preview module documents its own usage in `Dataset_Preview/README.md`; its automated tests are under `Dataset_Preview/tests/`.

## Data and artifact policy

LIDC-IDRI/DICOM data, annotations downloaded as dataset material, generated visualizations, processed arrays, checkpoints, model weights, caches and experiment logs stay outside Git. Do not commit patient-derived data or credentials. The existing local checkpoint is intentionally ignored and is not evidence of a validated result without provenance and reproducible metrics.

## Project documentation

- `docs/PROJECT_STATUS.md` — verified status and next actions.
- `docs/EXPERIMENTS.md` — reusable experiment record.
- `docs/DECISIONS.md` — architectural and repository decisions.
- `docs/THIRD_PARTY.md` — MambaVision upstream provenance and local modifications.
