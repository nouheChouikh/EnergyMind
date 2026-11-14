"""Checks that the detector can be imported and keeps its boundaries."""

import importlib  # imports a module from its name given as text

# The 7 folders of detector/, in plan order (steps 1 to 8).
FOLDERS = ["preparation", "rules", "columns", "forest", "diagnosis", "incidents", "feedback"]


def test_detector_and_its_folders_import():
    """detector and each of its folders can be imported."""
    # The library itself: fails if detector/ is not installed in .venv.
    importlib.import_module("detector")
    # Each folder, e.g. "detector.rules": fails if a folder is missing or has no __init__.py.
    for name in FOLDERS:
        importlib.import_module(f"detector.{name}")
