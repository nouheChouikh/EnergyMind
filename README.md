# EnergyMind

EnergyMind is a trust layer for a smart-metered neighbourhood. Every hour it checks each meter reading, tells a genuine grid event from a house problem, and gives its verdict with a confidence and a plain explanation. It compares every meter with its neighbours and with a reference meter at the transformer.

## Status

| Stage | What | Status |
|---|---|---|
| 1. Dataset | a simulated street built from real smart-meter data, with documented injected problems | done (tags `v1.0-phase1`, `v1.1-phase1`) |
| 2. Detection | rules, one losses regression, an Isolation Forest, explanations and diagnosis | next: designed in [`plan.md`](plan.md) |
| 3. Operator app | an Angular app where operators follow incidents and give their decisions | later |

## The dataset
- Real hourly consumption of Spanish households (GoiEner smart meter data), cleaned and assembled into one simulated low-voltage street.
- Simulated voltage, current and a reference meter at the transformer, computed with simple, stated physics.
- Temperature, weekends and holidays for every hour.
- Problems injected into the second year, each listed in an answer key so detection can be measured: real grid events, data problems, real changes in use that must not raise alarms, and one held-out type to test unseen problems.

Numbers, formulas and limits are generated with the data: see `data/DATA_CARD.md`, `data/RECIPES.md` and `data/checks_report.txt`.

**Grading files:** `data/answers.csv` (the answer key) and `data/normal/` (the same street without problems) are only for grading. The detection system never reads them.

## Repository
- `plan.md` - the detection design (stage 2): build from this file
- `data/` - the dataset, the answer key and the generated documentation
- `src/` - the code that builds the dataset (`config.py` holds every assumption, `run_all.py` runs everything)
- `notebooks/01_exploration.ipynb` - exploration with charts
- `figures/` - charts saved by the notebook

## Reproduce
Tested with Python 3.12:

    pip install -r requirements.txt
    python src/run_all.py

This rebuilds the dataset and runs the checks and the notebook; the same seed gives the same dataset. To redo the cleaning from the original archive, put `imp-pre.tzst` and `metadata.csv` from Zenodo record 7362094 in `raw/` and run `python src/run_all.py --from-raw`.

## Licences and credit
- Code: MIT (`LICENSE`).
- Data derived from GoiEner smart meters data (University of Deusto and GoiEner, doi:10.5281/zenodo.7362094), shared under CC BY-SA 4.0.
- Temperatures: Open-Meteo, CC BY 4.0.
