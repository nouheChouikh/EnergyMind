"""Input checks: stop early, with a clear message, if the inputs aren't what the detector expects (plan step 1).

What it does: three checks, run once when learning on a new site (not every hour):
  - every needed column is present (the error names the missing ones);
  - the units look right: voltages in volts and energy in kWh (not kilovolts, not Wh);
  - the transformer and the meters label the same hour.

Why it exists: a wrong unit or a one-hour shift between the two sources wouldn't crash anything. It would
quietly make every hour look strange, and the system would learn from bad data. Better to stop at once.

Choices:
  - Medians, not averages: a few broken readings can't move a median.
  - Same hour: the transformer and its homes' sum are closest when compared hour against the same hour.
    If they're closer with a one-hour shift, the two sources label hours differently
    (one marks the start of the hour, the other the end, or a clock is off).
"""

import pandas as pd  # tables

from detector import settings  # the expected ranges for voltage and energy


def check_required_columns(table, required, table_name):
    """Stops if the table lacks any of the required columns, and names them."""
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise ValueError(f"{table_name}: missing column(s) {', '.join(missing)}")


def check_units(home_hours):
    """Stops if the median voltage doesn't look like volts, or the median hourly energy doesn't look like kWh."""
    # Median voltage over every reading (empty readings are skipped).
    median_voltage = home_hours["voltage_v"].median()
    low, high = settings.VOLTAGE_MEDIAN_RANGE
    if not low <= median_voltage <= high:
        raise ValueError(
            f"median meter voltage is {median_voltage:.4g}, expected between {low:g} and {high:g}: "
            "are the voltages in volts?"
        )

    # Median energy per home and hour.
    median_kwh = home_hours["kwh"].median()
    low, high = settings.KWH_MEDIAN_RANGE
    if not low <= median_kwh <= high:
        raise ValueError(
            f"median hourly energy is {median_kwh:.4g}, expected between {low:g} and {high:g}: "
            "is the energy in kWh?"
        )


def gap_with_shift(phase_hours, shift_hours):
    """Average distance (kWh) between the transformer and its homes' sum, when each transformer reading is
    compared with the homes' sum shift_hours later (0: the same hour, -1: one hour earlier, 1: one hour later).
    """
    distances = []
    for _phase, one_phase in phase_hours.groupby("phase"):
        # One phase at a time, in time order: consecutive rows are consecutive hours.
        one_phase = one_phase.sort_values("timestamp")
        # shift(-n) puts on row t the homes' sum of hour t + n, so it sits next to the transformer of hour t.
        homes_shifted = one_phase["homes_kwh"].shift(-shift_hours)
        distances.append((one_phase["transformer_kwh"] - homes_shifted).abs())
    # Average over every phase and hour; empty values (e.g. the first or last hour after shifting) are skipped.
    return pd.concat(distances).mean()


def check_same_hour(phase_hours):
    """Stops if the transformer matches its homes better with a one-hour shift than at the same hour."""
    same_hour = gap_with_shift(phase_hours, 0)
    one_hour_earlier = gap_with_shift(phase_hours, -1)
    one_hour_later = gap_with_shift(phase_hours, 1)
    if same_hour >= min(one_hour_earlier, one_hour_later):
        raise ValueError(
            "the transformer matches the homes better with a one-hour shift than at the same hour: "
            "the two sources don't label hours the same way (start vs end of the hour, or a clock is off)"
        )
