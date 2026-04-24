from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"

OUTPUT_PLOTS = BASE_DIR / "outputs" / "plots"
OUTPUT_REPORTS = BASE_DIR / "outputs" / "reports"