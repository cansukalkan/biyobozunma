"""ONNX dışa aktarım testleri."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bozunma.train.export import export_onnx


def test_export_onnx_not_found(tmp_path: Path) -> None:
    non_existent = tmp_path / "missing.pt"
    with pytest.raises(FileNotFoundError):
        export_onnx(non_existent)


@patch("bozunma.train.export.YOLO")
def test_export_onnx_success(mock_yolo: MagicMock, tmp_path: Path) -> None:
    weights = tmp_path / "best.pt"
    weights.touch()

    expected_onnx = tmp_path / "best.onnx"
    expected_onnx.touch()

    fake_model = MagicMock()
    fake_model.export.return_value = str(expected_onnx)
    mock_yolo.return_value = fake_model

    result = export_onnx(weights, imgsz=640)
    assert result == expected_onnx
    fake_model.export.assert_called_once_with(format="onnx", imgsz=640)
