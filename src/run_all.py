"""Rebuild the whole Phase 1 dataset, checks and notebook.

    python src/run_all.py              uses the committed data/source_subset.parquet (no big download)
    python src/run_all.py --from-raw   redoes step 1 from raw/imp-pre.tzst and raw/metadata.csv
    python src/run_all.py --skip-notebook
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

    import step1_clean_select
    import step2_context
    import step3_simulate_normal
    import step4_inject_problems
    import step5_checks

    C.DATA_DIR.mkdir(exist_ok=True)
    if args.from_raw or not (C.DATA_DIR / "source_subset.parquet").exists():
        step("Step 1/6  clean GoiEner data and choose the homes", step1_clean_select.main)
    else:
        print("\n== Step 1/6  skipped: using data/source_subset.parquet (add --from-raw to redo it)")
    step("Step 2/6  context: temperature, weekends, holidays, periods", step2_context.main)
    step("Step 3/6  street wiring and normal values", step3_simulate_normal.main)
    step("Step 4/6  inject problems and write the answer file", step4_inject_problems.main)
    step("Step 5/6  checks and data card", step5_checks.main)
    if args.skip_notebook:
        print("\n== Step 6/6  notebook skipped")
    else:
        step("Step 6/6  run the exploration notebook (figures/)", run_notebook)
    print("\nDone. Read data/checks_report.txt, then commit.")


if __name__ == "__main__":
    main()
