from pathlib import Path

import pytest


@pytest.fixture
def carpeta_raw(tmp_path: Path) -> Path:
    """Carpeta vacía donde cada test escribe sus zips inventados."""
    carpeta = tmp_path / "raw"
    carpeta.mkdir()
    return carpeta
