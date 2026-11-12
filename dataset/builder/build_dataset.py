"""Rebuild the whole Phase 1 dataset, checks and exploration notebook. Run from the repository root:

    python dataset/builder/build_dataset.py                  uses the committed sources (no big download)
    python dataset/builder/build_dataset.py --from-raw       redoes step 1 from dataset/data/sources/goiener_download/
    python dataset/builder/build_dataset.py --skip-notebook
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C  # noqa: E402


def step(title, fn):
    print(f"\n== {title}", flush=True)
    t = time.time()
    fn()
    print(f"   ({time.time() - t:.0f} s)", flush=True)


def run_notebook():
    cmd = [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
           "--ExecutePreprocessor.timeout=1800", str(C.NOTEBOOK)]
    if subprocess.run(cmd).returncode != 0:
        print("   The notebook did not run. Fix: `python -m ipykernel install --user`, then rerun:\n   "
              + " ".join(cmd[1:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-raw", action="store_true", help="redo step 1 from the GoiEner archive")
    ap.add_argument("--skip-notebook", action="store_true")
    args = ap.parse_args()

    from steps import (step1_homes_selection, step2_context_building, step3_street_simulation,
                       step4_problems_injection, step5_dataset_checks)

    C.DATA_DIR.mkdir(exist_ok=True)
    if args.from_raw or not C.SELECTED_HOMES_READINGS.exists():
        step("Step 1/6  clean GoiEner data and choose the homes", step1_homes_selection.main)
    else:
        print("\n== Step 1/6  skipped: using dataset/data/sources/selected_homes_readings.parquet (add --from-raw to redo it)")
    step("Step 2/6  context: temperature, weekends, holidays, periods", step2_context_building.main)
    step("Step 3/6  street wiring and normal values", step3_street_simulation.main)
    step("Step 4/6  inject problems and write the answer key", step4_problems_injection.main)
    step("Step 5/6  checks and data card", step5_dataset_checks.main)
    if args.skip_notebook:
        print("\n== Step 6/6  notebook skipped")
    else:
        step("Step 6/6  run the exploration notebook (dataset/exploration/figures/)", run_notebook)
    print("\nDone. Read dataset/data/dataset_checks_report.txt, then commit.")


if __name__ == "__main__":
    main()
