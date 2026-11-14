# EnergyMind

EnergyMind is a trust layer for a smart-metered neighbourhood. Every hour it checks each meter reading, tells a genuine grid event from a house problem, and gives its verdict with a confidence and a plain explanation. It compares every meter with its neighbours and with a reference meter at the transformer.

## Status

| Stage | Folder | What | Status |
|---|---|---|---|
| 1. Dataset | `dataset/` | a simulated street built from real smart-meter data, with documented injected problems | done |
| 2. Detection | `detection/` | rules, one losses regression, an Isolation Forest, explanations and diagnosis | next: designed in [`detection/detection_plan.md`](detection/detection_plan.md) |
| 3. Operator app | `operator_app/` (created when the work starts) | an Angular app where operators follow incidents and give their decisions | later |

## Repository map

```
dataset/                  stage 1: the dataset
├── README.md             guide to this stage: how to rebuild it, what each step does
├── builder/              the code that builds data/ (config.py holds every setting and every file path)
├── data/                 the dataset itself; its README.md is the data card
│   ├── system_inputs/    what a utility would send: the only files the detection system reads
│   ├── grading/          answer key and problem-free copies: for scoring only
│   └── sources/          what the dataset is built from
└── exploration/          the exploration notebook and its charts
detection/                stage 2: the detection system
└── detection_plan.md     the design to build from
```

## The dataset
- Real hourly consumption of Spanish households (GoiEner smart meter data), cleaned and assembled into one simulated low-voltage street.
- Simulated voltage, current and a reference meter at the transformer, computed with simple, stated physics.
- Temperature, weekends and holidays for every hour.
- Problems injected into the second year, each listed in an answer key so detection can be measured: real grid events, data problems, real changes in use that must not raise alarms, and one held-out type to test unseen problems.

Numbers, formulas and limits are generated with the data: see the data card [`dataset/data/README.md`](dataset/data/README.md), the recipes `dataset/data/grading/problem_recipes.md` and the checks `dataset/data/dataset_checks_report.txt`.

**Grading files:** `dataset/data/grading/` holds the answer key and the same street without problems. These files are only for grading; the detection system never reads them.

## Reproduce the dataset
Tested with Python 3.12. From the repository root:

    pip install -r dataset/builder/requirements.txt
    python dataset/builder/build_dataset.py

This rebuilds the dataset and runs the checks and the notebook; the same seed gives the same dataset. To redo the cleaning from the original archive, put `imp-pre.tzst` and `metadata.csv` from Zenodo record 7362094 in `dataset/data/sources/goiener_download/` and run `python dataset/builder/build_dataset.py --from-raw`. Details in [`dataset/README.md`](dataset/README.md).

## Licences and credit
- Code: MIT (`LICENSE`).
- Data derived from GoiEner smart meters data (University of Deusto and GoiEner, doi:10.5281/zenodo.7362094), shared under CC BY-SA 4.0.
- Temperatures: Open-Meteo, CC BY 4.0.
