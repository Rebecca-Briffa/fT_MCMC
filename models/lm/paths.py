from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"


def data_file(*parts: str) -> str:
    return str(DATA_DIR.joinpath(*parts))
