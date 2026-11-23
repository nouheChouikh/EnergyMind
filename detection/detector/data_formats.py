"""What goes in (readings) and what comes out (findings, later incidents and cards).

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
#   transformer_kwh, transformer_voltage_v   what the transformer measured on this phase
#   homes_kwh                                sum of the kWh reported by this phase's homes
#   nb_homes_reporting                       number of this phase's homes that sent a reading
#   nb_homes_on_phase                        number of homes on this phase
PHASE_HOURS_COLUMNS = [
    "timestamp", "phase", "transformer_kwh", "transformer_voltage_v",
    "homes_kwh", "nb_homes_reporting", "nb_homes_on_phase",
]

# ----------------------------------------------------------------------------- results
# Rule findings (plan step 2): one row per finding.
#   scope     where it sits: home, phase, street or transformer (names in settings.py)
#   phase     the phase concerned (for a home: its phase; for the transformer: the phase whose reading
#             is broken); empty only for the whole street
#   home_id   the home concerned; empty unless the scope is home
#   verdict   the verdict name (names in settings.py)
RULE_FINDINGS_COLUMNS = ["timestamp", "scope", "phase", "home_id", "verdict"]
