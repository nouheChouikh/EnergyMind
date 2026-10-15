# EnergyMind - Phase 1: dataset and exploration

EnergyMind is a trust layer for a smart-metered neighbourhood. It checks every meter against its neighbours and against a reference meter at the transformer, tells real grid events apart from data problems, and explains each alert in plain language.

This repository holds the Phase 1 deliverables: the dataset, the code that builds it, and a first exploration.

## The dataset
- Real hourly consumption of Spanish households (GoiEner smart meter data), cleaned and assembled into one simulated low-voltage street.
- Simulated voltage, current and a reference meter at the transformer, computed with simple, stated physics.
- Temperature, weekends and holidays for every hour.
- Problems injected into the second year, each listed in an answer key so detection can be measured: real grid events, data problems, real changes in use that must not raise alarms, and one held-out type to test unseen problems.

Numbers, formulas and limits are generated with the data: see `data/DATA_CARD.md`, `data/RECIPES.md` and `data/checks_report.txt`.

## Repository
- `data/` - the dataset, the answer key and the generated documentation
- `src/` - the code (`config.py` holds every assumption, `run_all.py` runs everything)
- `notebooks/01_exploration.ipynb` - exploration with charts
- `figures/` - charts saved by the notebook

## Reproduce
Python 3.10 to 3.13:

    pip install -r requirements.txt
    python src/run_all.py

This rebuilds the dataset and runs the checks and the notebook; the same seed gives the same dataset. To redo the cleaning from the original archive, put `imp-pre.tzst` and `metadata.csv` from Zenodo record 7362094 in `raw/` and run `python src/run_all.py --from-raw`.

## Licences and credit
- Code: MIT (`LICENSE`).
- Data derived from GoiEner smart meters data (University of Deusto and GoiEner, doi:10.5281/zenodo.7362094), shared under CC BY-SA 4.0.
- Temperatures: Open-Meteo, CC BY 4.0.
