"""Checks that the detector can be imported and keeps its boundaries."""

import importlib  # imports a module from its name given as text
import re  # searches text with patterns
from pathlib import Path  # file paths that work on Windows and Linux

# The 7 folders of detector/, in plan order (steps 1 to 8).
FOLDERS = ["preparation", "rules", "features", "forest", "diagnosis", "incidents", "feedback"]

# detector/, found from this file's place: tests/ and detector/ sit side by side in detection/.
DETECTOR_DIR = Path(__file__).resolve().parents[1] / "detector"


def offenders(pattern, skip=()):
    """Files under detector/ whose text matches the pattern (files named in skip are not checked)."""
    found = []
    # Every .py file in detector/ and its folders, sorted so a failure always lists them in the same order.
    for path in sorted(DETECTOR_DIR.rglob("*.py")):
        if path.name in skip:
            continue
        # re.M: "^" matches at the start of every line, not only at the start of the file.
        if re.search(pattern, path.read_text(encoding="utf-8"), re.M):
            found.append(str(path.relative_to(DETECTOR_DIR)))
    return found


def test_detector_and_its_folders_import():
    """detector and each of its folders can be imported."""
    # The library itself: fails if detector/ is not installed in .venv.
    importlib.import_module("detector")
    # Each folder, e.g. "detector.rules": fails if a folder is missing or has no __init__.py.
    for name in FOLDERS:
        importlib.import_module(f"detector.{name}")


def test_detector_never_opens_data_files():
    """Only trained_model.py may write and read files (the trained model); everything else receives tables."""
    # open(), pandas readers and writers, and joblib (the tool that saves the forest).
    pattern = r"\bopen\(|read_csv|read_parquet|to_csv\(|to_parquet\(|joblib\.(dump|load)"
    assert offenders(pattern, skip={"trained_model.py"}) == []


def test_detector_never_mentions_grading_files():
    """The answer key and the problem-free copies are for scoring only: the detector must not know them."""
    assert offenders(r"grading|answer_key") == []


def test_detector_never_uses_the_period_column():
    """The period column tells practice from final test, and STEG doesn't send it."""
    # The word between quotes, i.e. used as a column name.
    assert offenders(r"[\"']period[\"']") == []


def test_detector_never_imports_evaluation():
    """Evaluation uses the detector, never the other way round."""
    # A line starting with "from evaluation", "import evaluation", "from shared" or "import shared".
    assert offenders(r"^\s*(from|import)\s+(evaluation|shared)\b") == []
