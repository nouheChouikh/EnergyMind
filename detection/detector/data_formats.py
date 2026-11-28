"""What goes in (readings, the memory), what learning keeps (the losses fit), what comes out (findings, incidents).

Each list names the columns the detector needs from a table, in order. The files a utility sends
(here, the dataset's system_inputs files) may hold more columns: the detector ignores them.
"""

# ----------------------------------------------------------------------------- input tables
# Meter readings: one row per home and hour. estimated = 1 when the utility filled the value in itself.
# Needs every column of the dataset's meter readings files.
METER_READINGS_COLUMNS = ["home_id", "timestamp", "kwh", "voltage_v", "current_a", "estimated"]

# Transformer readings: one row per hour, energy (kWh) and voltage (V) on each phase.
# Left out: energy_kwh_total (the street total: the detector works phase by phase).
TRANSFORMER_READINGS_COLUMNS = [
    "timestamp",
    "energy_kwh_A", "energy_kwh_B", "energy_kwh_C",
    "voltage_v_A", "voltage_v_B", "voltage_v_C",
]

# Street wiring: one row per home (its cable, its phase, its contracted power in kW).
# Left out: position and distance_m (neighbours are found by cable and phase),
# province, zip_code and goiener_user (where the home's data came from: not needed to detect).
STREET_WIRING_COLUMNS = ["home_id", "cable", "phase", "contracted_kw"]

# Hourly context: one row per hour (weekend and public holiday flags, 1 or 0).
# Left out: temperature_c (weather is a real change in use, not a fault: the usual follows the season, plan step 4),
# holiday_name (the flag is enough), and period (dataset bookkeeping that tells practice from final test:
# a utility never sends it, and the detector must never use it).
HOURLY_CONTEXT_COLUMNS = ["timestamp", "is_weekend", "is_holiday"]

# ----------------------------------------------------------------------------- tables built by preparation (plan step 1)
# Home table: one row per home and hour, for every home in the wiring and every hour in the context.
# A reading that never arrived is a row with kwh, voltage_v and current_a empty.
HOME_HOURS_COLUMNS = [
    "timestamp", "home_id", "phase", "cable", "contracted_kw", "day_type",
    "kwh", "voltage_v", "current_a", "estimated",
]

# Phase table: one row per phase and hour.
#   day_type                                 the hour's type of day (the same for every phase)
#   transformer_kwh, transformer_voltage_v   what the transformer measured on this phase
#   homes_kwh                                sum of the kWh reported by this phase's homes
#   nb_homes_reporting                       number of this phase's homes that sent a reading
#   nb_homes_on_phase                        number of homes on this phase
PHASE_HOURS_COLUMNS = [
    "timestamp", "phase", "day_type", "transformer_kwh", "transformer_voltage_v",
    "homes_kwh", "nb_homes_reporting", "nb_homes_on_phase",
]

# ----------------------------------------------------------------------------- the forest table, built each hour (plan step 4)
# The five features: the values the forest learns from and scores, in the plan's order.
FOREST_FEATURES = [
    "consumption_vs_usual",  # the home's kWh ÷ its usual kWh
    "voltage_gap_with_neighbours_vs_usual",  # the home's voltage gap with its neighbours − its usual gap (V)
    "phase_unexplained_energy",  # the % of its phase's energy that nothing explains (its phase's value)
    "transformer_voltage_vs_usual",  # its phase's transformer voltage − its usual (V) (its phase's value)
    "hour",  # 0 to 23
]
# The forest table: one row per home at the new hour. timestamp and home_id are its name tag: the forest is given
# only the five features, but a flag must say which home and which hour it's about.
FOREST_TABLE_COLUMNS = ["timestamp", "home_id", *FOREST_FEATURES]

# ----------------------------------------------------------------------------- the memory the app sends each hour (plan step 9)
# Past home rows and past phase rows (about the last 28 days): the same columns as the two tables above, plus each
# row's trust state, decided when its hour was checked (trusted, untrusted or unchecked: names in settings.py).
# Home rows also keep their voltage gap with their neighbours, computed when their hour was checked: the gap's usual
# is the median of past gaps, so each hour's gap is kept, like its trust.
HOME_MEMORY_COLUMNS = [*HOME_HOURS_COLUMNS, "voltage_gap_with_neighbours", "trust"]
PHASE_MEMORY_COLUMNS = [*PHASE_HOURS_COLUMNS, "trust"]  # the phase table's columns, then "trust"

# Last usuals: for each home (or phase), hour of day and type of day, the last usual that could be computed. It's
# the fallback when fewer than 3 trusted values remain (the memory only keeps about 28 days). Empty at the start;
# a new usual overwrites the old value, so it holds at most one value per home (or phase), hour and type of day.
#   usual_of        which usual: consumption (a home's kWh) or transformer_voltage (a phase's voltage)
#   home_or_phase   the home id (e.g. H07), or the phase (A, B or C) for the transformer voltage
#   hour_of_day     0 to 23
#   day_type        weekday or weekend_or_holiday
#   usual           the last usual computed
LAST_USUALS_COLUMNS = ["usual_of", "home_or_phase", "hour_of_day", "day_type", "usual"]

# ----------------------------------------------------------------------------- learned once, at learning
# Losses fit (plan step 3): one row per phase, kept in the trained model and used every hour.
#   a, b   expected cable losses = a + b x E², where E is the transformer's energy on the phase (kWh)
LOSSES_FIT_COLUMNS = ["phase", "a", "b"]

# ----------------------------------------------------------------------------- results
# Rule findings (plan step 2): one row per finding.
#   scope     where it sits: home, phase, street or transformer (names in settings.py)
#   phase     the phase concerned (for a home: its phase; for the transformer: the phase whose reading
#             is broken); empty only for the whole street
#   home_id   the home concerned; empty unless the scope is home
#   verdict   the verdict name (names in settings.py)
RULE_FINDINGS_COLUMNS = ["timestamp", "scope", "phase", "home_id", "verdict"]
