# Stage 1: the dataset

This folder builds and documents the EnergyMind street dataset: the real hourly consumption of 75 Spanish homes (GoiEner), assembled into one simulated low-voltage street, with problems injected into 2019 and listed in an answer key.

## What's here

| Folder | Content |
|---|---|
| `builder/` | the code that builds `data/` |
| `data/` | the dataset itself; its `README.md` is the data card (sources, method, files, checks, limits) |
| `exploration/` | `dataset_exploration.ipynb`, a first look at the data, and the charts it saves in `figures/` |

Inside `data/`:
- `system_inputs/`: what a utility would send (meter readings, transformer readings, street wiring, hourly context). The only files the detection system may read.
- `grading/`: the answer key, the problem recipes and the same street without problems. For scoring only.
- `sources/`: what the dataset is built from. `sources/goiener_download/` holds the 791 MB GoiEner download when step 1 is redone; it is never committed.

## Rebuild

From the repository root (tested with Python 3.12.10 and the versions pinned in `builder/requirements.txt`):

    pip install -r dataset/builder/requirements.txt
    python dataset/builder/build_dataset.py                    # steps 2 to 6 (step 1 reuses the saved selection)
    python dataset/builder/build_dataset.py --skip-notebook    # same, without rerunning the notebook
    python dataset/builder/build_dataset.py --from-raw         # also redo step 1 from the GoiEner download

The seed is fixed (2026), so a rebuild gives exactly the same files. Every setting and every file path is in `builder/config.py`.

## The steps

| Step | File (in `builder/steps/`) | What it does | Writes (in `dataset/`) |
|---|---|---|---|
| 1 | `step1_homes_selection.py` | cleans the GoiEner data and draws the 75 homes; runs only with `--from-raw`, or when its output is missing | `data/sources/selected_homes_readings.parquet`, `data/sources/selected_homes.csv`, `data/sources/homes_selection_log.csv` |
| 2 | `step2_context_building.py` | temperature, weekend, holiday and period of every hour | `data/system_inputs/hourly_context.csv` (and `data/sources/openmeteo_temperatures.csv` if the temperatures have to be downloaded) |
| 3 | `step3_street_simulation.py` | places the homes on the street and simulates voltage, current, cable losses and the transformer, with no problem anywhere | `data/system_inputs/street_wiring.csv`, `data/grading/normal_meter_readings.parquet`, `data/grading/normal_transformer_readings.parquet` |
| 4 | `step4_problems_injection.py` | plans the problems and injects them into 2019 | `data/system_inputs/meter_readings_2018.parquet`, `data/system_inputs/meter_readings_2019.parquet`, `data/system_inputs/transformer_readings.parquet`, `data/grading/answer_key.csv`, `data/grading/problem_recipes.md` |
| 5 | `step5_dataset_checks.py` | checks the dataset and writes its documentation | `data/dataset_checks_report.txt`, `data/README.md` (the data card) |
| 6 | (inside `build_dataset.py`) | reruns the exploration notebook | the notebook's results and `exploration/figures/` |

The steps share code from `builder/shared/`:
- `street_physics.py`: the street's wiring, physics and measurement noise, and how each problem changes the readings;
- `problems_planning.py`: where, when and how big each problem is, and the answer key;
- `helpers.py`: the time axis, random streams, file reading and writing, fingerprints.

## Run one step on its own

From the `dataset/builder/` folder, for example:

    python -m steps.step2_context_building

Each step reads the files written by the steps before it.
