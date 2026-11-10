# -*- coding: utf-8 -*-
"""
tests/test_notebooks.py - Kiểm thử tính hợp lệ và độc lập cấu trúc của 6 Jupyter Notebooks.

Yêu cầu kiểm thử:
  - 6 notebooks tồn tại tại đúng vị trí: <arch>/notebooks/{colab,kaggle,local}_train.ipynb.
  - Mỗi notebook là một file JSON hợp lệ theo chuẩn nbformat 4.
  - unet_base notebooks không chứa bất kỳ từ khóa hay lệnh gọi mamba_unet nào (R1).
  - mamba_unet notebooks không chứa bất kỳ từ khóa hay lệnh gọi unet_base nào (R1).
"""

import json
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

NOTEBOOK_PATHS = [
    _ROOT / "unet_base" / "notebooks" / "colab_train.ipynb",
    _ROOT / "unet_base" / "notebooks" / "kaggle_train.ipynb",
    _ROOT / "unet_base" / "notebooks" / "local_train.ipynb",
    _ROOT / "mamba_unet" / "notebooks" / "colab_train.ipynb",
    _ROOT / "mamba_unet" / "notebooks" / "kaggle_train.ipynb",
    _ROOT / "mamba_unet" / "notebooks" / "local_train.ipynb",
]


class TestNotebooks(unittest.TestCase):
    def test_all_six_notebooks_exist_and_valid_json(self):
        """Đảm bảo đủ 6 notebook và cú pháp JSON chuẩn nbformat 4."""
        for nb_path in NOTEBOOK_PATHS:
            self.assertTrue(nb_path.exists(), f"Không tìm thấy notebook: {nb_path}")
            with open(nb_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data.get("nbformat"), 4, f"nbformat phải là 4: {nb_path}")
            self.assertIn("cells", data, f"Notebook thiếu cells: {nb_path}")
            self.assertGreaterEqual(len(data["cells"]), 3, f"Số cell quá ít: {nb_path}")

    def test_notebook_architecture_isolation(self):
        """Đảm bảo không rò rỉ mã chéo giữa 2 kiến trúc trong các notebook (R1)."""
        for nb_path in NOTEBOOK_PATHS:
            arch = nb_path.parent.parent.name
            with open(nb_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            content = "".join("".join(c.get("source", [])) for c in data["cells"])
            if arch == "unet_base":
                self.assertNotIn(
                    "mamba_unet",
                    content,
                    f"Notebook của unet_base ({nb_path.name}) chứa tham chiếu tới mamba_unet!",
                )
            elif arch == "mamba_unet":
                self.assertNotIn(
                    "unet_base",
                    content,
                    f"Notebook của mamba_unet ({nb_path.name}) chứa tham chiếu tới unet_base!",
                )


if __name__ == "__main__":
    unittest.main()
