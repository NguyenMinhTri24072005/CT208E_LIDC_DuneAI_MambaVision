# Third-party Dependencies

## MambaVision

- **Upstream repository:** https://github.com/NVlabs/MambaVision.git
- **Original branch:** `main`
- **Original commit:** `7860a506b2eb844eaaae676f08461ce8c3c26f43`
- **Imported date:** 2026-09-16 (date converted to vendored source and provenance recorded; original clone date: Cần xác nhận)
- **License:** Preserved at `LIDC_MambaVision_Project/third_party/MambaVision/LICENSE`.
- **Local modifications:**
  - `mambavision/models/mamba_vision.py`
  - `mambavision/models/registry.py`
- **Modification purpose:**
  - Cần xác nhận.
  - The diff adds compatibility paths for multiple `timm` APIs, explicit checkpoint loading behavior compatible with different PyTorch signatures, and a pure-PyTorch selective-scan fallback when `mamba_ssm` cannot be imported.
  - The original intent, correctness, performance characteristics, and validation status of these local changes are not established by the available Git history.
- **Integration method:** Vendored source under `LIDC_MambaVision_Project/third_party/MambaVision`.

### Local Diff Summary at Import

- `mambavision/models/mamba_vision.py`: 90 insertions and 11 deletions. It changes imports and compatibility handling, adds the selective-scan fallback, and changes checkpoint loading to request `weights_only=False` with a fallback for older signatures.
- `mambavision/models/registry.py`: 4 insertions and 1 deletion. It changes checkpoint loading to request `weights_only=False` with a fallback for older signatures.
- Combined diff: 94 insertions and 12 deletions across 2 files.
- Untracked runtime artifacts present before cleanup: `mambavision/__pycache__/` and `mambavision/models/__pycache__/`. These are not part of the upstream source provenance.

### Upstream History Snapshot

```text
7860a50 Update mamba_vision.py
3945e9f Update LICENSE
7a3f8ac Update README.md
42a4026 Update mamba_vision.py
4286a73 Update README.md
```

### Verification Artifacts

- Exact local patch: `docs/third_party_backups/MambaVision-local-changes-7860a506-2026-09-16.patch`
- Pre-cleanup file manifest: `docs/third_party_backups/MambaVision-files-before-cleanup-2026-09-16.json`
- Patch SHA-256: `4a0171f03248527710d16444c067514e24425e4baa9ae69a78959456fc7760de`

The patch is retained only as a provenance and recovery artifact. It was not applied during vendoring because the working tree already contained these local changes.
