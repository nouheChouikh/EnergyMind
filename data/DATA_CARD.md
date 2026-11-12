# Data card - EnergyMind street dataset (Phase 1)

A simulated low-voltage street built from the **real hourly consumption of 75 Spanish homes** (GoiEner smart meter data, 2018-2019). Each home has a meter that reports energy, voltage and current every hour; a reference meter at the transformer measures energy and voltage on each of the 3 phases. In 2019, 125 problems are injected following written recipes, and every one of them is listed in `answers.csv`.

Everything here is rebuilt by `python src/run_all.py` (fixed seed 2026); every invented number is in `src/config.py`.

## Source

- **Energy:** GoiEner smart meters data, `imp-pre` subset (hourly kWh before March 2020, with an `imputed` flag). Quesada, Borges, Astigarraga & Merveille, Zenodo, doi:10.5281/zenodo.7362094; data paper: Scientific Data 11, 59 (2024), doi:10.1038/s41597-023-02846-0. Timestamps are Spanish wall-clock time; the authors filled the skipped spring hour and averaged the repeated autumn hour.
- **Temperature:** Open-Meteo historical weather archive (ERA5 reanalysis), hourly 2 m temperature at 43.3183, -1.9812 (capital of the province), CC BY 4.0.
- **Holidays:** Spanish national and regional calendar from the `holidays` Python package (24 holidays in 2018-2019).

## How the homes were chosen (cleaning)

| Stage | Rule | Homes left |
|---|---|---|
| metadata | all series in metadata.csv | 25559 |
| metadata | household (CNAE 98xx) | 20794 |
| metadata | no self-consumption (no solar panels) | 20655 |
| metadata | household tariff (2.x, up to 15 kW) | 20477 |
| metadata | series dates cover 2018-2019 | 6950 |
| readings | series found in imp-pre.tzst | 6923 |
| readings | complete hourly 2018-2019 (gaps up to 48 h filled, flagged estimated) | 6899 |
| readings | estimated hours at most 5% | 6858 |
| readings | yearly use 600-12000 kWh in 2018 and in 2019 | 6193 |
| readings | 2019 vs 2018 change at most 35% | 5840 |
| readings | no unexplained zeros (runs up to 0 h, at most 0% of hours) | 2965 |
| readings | no flat stretches (at most 3 identical measured readings in a row) | 1336 |
| selection | in Gipuzkoa, the province with most candidates | 588 |
| selection | random draw with seed 2026 | 75 |

Result: 75 homes from **Gipuzkoa**, 14 postcodes (56 homes have no postcode: GoiEner removes postcodes of towns under 50,000 people). Homes with more than 48 missing hours were removed. Up to 48 missing hours would be filled by linear interpolation and flagged as estimated, but none of the 75 chosen homes has a missing hour; their only estimated hours are the clock-change hours flagged by GoiEner.

## The street

- 3 cables leave the transformer; homes are spaced 15 m apart (farthest home 375 m away).
- Phases rotate A, B, C along each cable, starting on a different phase per cable: A = 25 homes, B = 25 homes, C = 25 homes.
- Homes are placed on cables at random (fixed seed). They were not neighbours in reality.

## Simulated values (formulas)

| Value | How it is computed |
|---|---|
| Energy (kWh) | the real GoiEner reading of the home |
| Current (A) | energy as average power / (230 V x 0.95), plus 0.5% meter noise |
| Transformer voltage (V) | 233 V minus up to 3 V at the street's peak load, plus 0.2 V noise |
| Home voltage (V) | transformer voltage minus the cable drop: R x sum over homes on the same cable and phase of I x min(their distance, this distance), R = 0.64 ohm/km (phase + neutral); plus 0.3 V noise |
| Cable losses (kWh) | sum over cable segments of R x length x (current through the segment)^2 |
| Transformer energy (kWh) | per phase: true energy of its homes + cable losses, plus 0.25% reference-meter noise |

## Periods

| Period | Months | Problems injected |
|---|---|---|
| learning | 2018: all months | 0 |
| practice | 2019: Jan, Mar, May, Jul, Sep, Nov | 57 |
| final_test | 2019: Feb, Apr, Jun, Aug, Oct, Dec | 68 |

## Injected problems

| Type | Practice | Final test |
|---|---|---|
| `outage_street` | 3 | 3 |
| `missing_all` | 3 | 3 |
| `outage_phase` | 4 | 4 |
| `voltage_sag` | 3 | 3 |
| `voltage_surge` | 3 | 3 |
| `missing_one` | 6 | 6 |
| `theft` | 9 | 9 |
| `over_reporting` | 0 | 9 |
| `voltage_drift` | 6 | 6 |
| `impossible_value` | 6 | 6 |
| `stuck_meter` | 6 | 6 |
| `holiday_absence` | 6 | 6 |
| `new_ac` | 2 | 4 |

Recipes and rules: `RECIPES.md`. The answer key: `answers.csv`.

## Files

| File | Size | Content |
|---|---|---|
| `readings_2018.parquet` | 2.9 MB | learning year: one row per home and hour - kWh, voltage_v, current_a, estimated |
| `readings_2019.parquet` | 3.0 MB | practice + final test, same columns, with the injected problems |
| `transformer.parquet` | 0.4 MB | reference meter: energy and voltage per phase each hour, plus the street total |
| `wiring.csv` | 0.0 MB | cable, phase, position and distance of each home |
| `context.csv` | 0.7 MB | temperature, weekend, holiday and period of each hour |
| `answers.csv` | 0.0 MB | every injected problem: type, homes, start, end, size, expected verdict, recipe |
| `RECIPES.md` | 0.0 MB | what each problem type changes, sizes and counts |
| `normal/readings_normal.parquet` | 6.9 MB | the same street with no problem at all (simulated normal values) |
| `normal/transformer_normal.parquet` | 0.4 MB | the transformer with no problem at all |
| `source_subset.parquet` | 4.3 MB | original GoiEner readings of the 75 homes, 2018-2019 (rebuild without the 791 MB download) |
| `homes.csv` | 0.0 MB | the chosen homes: GoiEner id, province, postcode, contracted power, yearly use |
| `selection_log.csv` | 0.0 MB | how many homes each cleaning rule removed |
| `weather_openmeteo.csv` | 0.7 MB | raw hourly temperature download (UTC) |
| `checks_report.txt` | 0.0 MB | results of the checks below |

## Checks (full numbers in `checks_report.txt`)

- **PASS** Rebuilding from the same seed gives exactly the same dataset
- **PASS** 2018 (learning) is identical to the normal simulation: no problem injected
- **PASS** Every planned problem was placed
- **PASS** Placement rules: no estimated hours, one month, own period, no overlap
- **PASS** Every injected problem changed the readings
- **PASS** Street balance on clean 2018 data: transformer = homes + cable losses + meter noise
- **PASS** Normal voltages stay inside 230 V +/- 10 % (EN 50160)
- **INFO** Honest meters repeating a value by coincidence (2018)
- **PASS** No unexplained zeros or flat stretches in the chosen homes
- **INFO** Hours estimated by GoiEner (flag `estimated`)
- **PASS** Every data file is under 50 MB (GitHub limit 100 MB)

## Known limits

- Voltage, current, cable losses and the transformer meter are simulated with the simple formulas above, not measured. Results on them show how a method behaves under these assumptions, not on a real network.
- The smallest theft that can be detected depends on the assumed meter noise and cable losses; Phase 2 reports results at several noise levels.
- The energy readings are real, but the homes were not neighbours: they come from one province and are assembled into one street.
- Injected problems follow clean recipes; real faults are messier and can overlap.
- One weather point for the whole province.

## Licences and credit

- Data derived from GoiEner smart meters data (University of Deusto and GoiEner), published here under CC BY-SA 4.0. Zenodo lists the source under CC BY 4.0 (its description also says CC-BY-SA); CC BY-SA 4.0 respects both.
- Temperatures: Open-Meteo (open-meteo.com), CC BY 4.0, from Copernicus ERA5.
- Code: MIT licence (see `LICENSE`).
