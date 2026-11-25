# Stage 2: the detection system

This folder builds EnergyMind's detection system and measures how right it is on the stage 1 dataset. It hands two things to the operator app (stage 3): the detector's functions and the saved model. It never runs live itself: the operator app's backend will call it once per hour.

The design (what the system does and why) is in [`detection_plan.md`](detection_plan.md).

## What's here

| Folder or file | Content |
|---|---|
| `detector/` | the detection system: a library used by evaluation now, and by the operator app's backend later. It never opens data files: callers pass tables in and get results back |
| `evaluation/` | the three scripts that run the detector on the dataset (learning on 2018, practice and final test on 2019), and the code they share in `shared/` |
| `outputs/` | what the scripts generate: the saved model and the two reports. Never edited by hand (see [`outputs/README.md`](outputs/README.md)) |
| `tests/` | automatic checks of the code, on small made-up data |
| `detection_plan.md` | the design |
| `requirements.txt` | the exact library versions used |
| `pyproject.toml` | makes `detector/` installable as a library |

## Layout

```
detection/
├── detector/                       the detection system
│   ├── __init__.py                 front door: the functions callers use
│   ├── settings.py                 every value chosen by people
│   ├── data_formats.py             what goes in (one hour of readings) and what comes out
│   ├── memory.py                   what is carried from one hour to the next
│   ├── trained_model.py            what learning produced, plus what is stored with it; save and load
│   ├── learning.py                 learn(): normal history → trained model
│   ├── hourly_check.py             start_memory() and check_hour(): one hour → results
│   ├── trust.py                    which home and phase rows may feed "usual"
│   ├── preparation/                tables.py, input_checks.py
│   ├── rules/                      meter_rules.py, street_rules.py, findings.py (all rules, in order)
│   ├── features/                   losses_regression.py, usual.py, forest_table.py (the five features)
│   ├── forest/                     forest_training.py, forest_scoring.py
│   ├── diagnosis/                  shap_reasons.py, verdict_patterns.py, home_naming.py
│   ├── incidents/                  incident_tracking.py, confidence.py, explanation_cards.py
│   └── feedback/                   operator_decisions.py, incident_records.py
├── evaluation/
│   ├── run_learning.py             learns from 2018 → outputs/saved_model/
│   ├── run_practice.py             practice months → outputs/practice_results/
│   ├── run_final_test.py           final-test months → outputs/final_test_results/ (once)
│   └── shared/                     paths, dataset_files, replay, answer_matching, measures,
│                                   threshold_baseline, report_writing
├── outputs/                        generated: saved_model/, practice_results/, final_test_results/
└── tests/                          conftest.py (shared test data), test_boundaries.py, and one test file
                                    per part: preparation, rules, features, trust, forest, diagnosis,
                                    incidents, feedback, hourly_check, answer_matching
```

## Where each plan step lives

| Plan step | Where |
|---|---|
| 1. Prepare the data | `detector/preparation/` |
| 2. Rules for definitions | `detector/rules/` |
| 3. The losses regression | `detector/features/losses_regression.py` |
| 4. The five features | `detector/features/`; the trust rule in `detector/trust.py` |
| 5. Train the Isolation Forest | `detector/forest/` |
| 6. Explain and diagnose each flag | `detector/diagnosis/` |
| 7. Incidents, scores, explanation card | `detector/incidents/` |
| 8. Operators and feedback | `detector/feedback/` |
| 9. The hourly loop | `detector/hourly_check.py` handles one hour; the loop over hours is `evaluation/shared/replay.py` now, the operator app's backend later |
| 10. Learning, practice, final test | `detector/learning.py` and the three `evaluation/run_*.py` scripts |
| 11. Later improvements | not built yet |

## Rules

1. The detector never opens a data file. Only `detector/trained_model.py` writes and reads files: the saved model, in a folder the caller chooses.
2. The detector never sees the grading files (answer key, problem-free copies) and never uses the `period` column, which STEG doesn't send. `evaluation/shared/dataset_files.py` removes that column before feeding the detector.
3. Only `evaluation/` reads `dataset/data/`. The answer key is read by `dataset_files.py` and used only by `answer_matching.py`.
4. Values chosen by people live in `detector/settings.py`; values learned from data live in the saved model.
5. Evaluation uses the detector, never the other way round.
6. Generated files live in `outputs/`, never next to the code.
7. The final test runs once and changes nothing, not even the confidence: `run_final_test.py` refuses to run if `outputs/final_test_results/` already exists.

`tests/test_boundaries.py` checks rules 1, 2 and 5 automatically.

## Set up (once)

From the repository root, with `.venv` active:

    pip install -r detection/requirements.txt    the exact library versions
    pip install -e detection                     makes `import detector` work; code edits apply immediately

## Run

    python detection/evaluation/run_learning.py      learns from 2018 → outputs/saved_model/
    python detection/evaluation/run_practice.py      practice months → outputs/practice_results/ (as often as needed)
    python detection/evaluation/run_final_test.py    final-test months → outputs/final_test_results/ (once, at the very end)
    python -m pytest detection/tests                 the automatic checks

Every pull request into `main` also runs these tests on GitHub (the "Detection tests" job in `.github/workflows/ci.yml`).

## Status

Two words, used the same way everywhere: a **step** is one of the 11 steps of `detection_plan.md` (what the system does); a **part** is one row of the build order below (what gets built, in which order). One part can cover several steps.

| Part | What is built | Plan steps | Where | Status |
|---|---|---|---|---|
| 1 | preparation and rules | 1, 2 | `detector/preparation/`, `detector/rules/` | **built** |
| 2 | losses, trust, usual, the five features | 3, 4 | `detector/features/`, `detector/trust.py` | in progress |
| 3 | the forest | 5 | `detector/forest/` | |
| 4 | learning and the first saved model | 10 (learning) | `detector/learning.py`, `detector/trained_model.py`, `evaluation/run_learning.py` | |
| 5 | diagnosis | 6 | `detector/diagnosis/` | |
| 6 | incidents, confidence, cards | 7 | `detector/incidents/` | |
| 7 | memory and the hourly check | 9 | `detector/memory.py`, `detector/hourly_check.py` | |
| 8 | practice | 10 (practice) | `evaluation/run_practice.py` | |
| 9 | feedback | 8 | `detector/feedback/` | |
| 10 | the final test, at the very end | 10 (final test) | `evaluation/run_final_test.py` | |

Part 1 is built and tested: the home and phase tables, the input checks, and the definition rules, combined by `rules/findings.py` into one table of findings per hour. Every rule is built for live use: it judges the new hour from the new hour plus the memory. The `evaluation/` scripts don't do anything yet; they start working with part 4.
