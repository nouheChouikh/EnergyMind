"""The home table and the phase table: the two tables every later step reads (plan step 1).

What it does: turns the utility's four inputs into
  - the home table: one row per home and hour, with the home's phase, cable, contracted power,
    the hour's type of day and the reading;
  - the phase table: one row per phase and hour, with what the transformer measured next to the sum
    of what the phase's homes reported, how many homes reported, and the hour's type of day.

Why it exists: every rule and column needs the same joins (which phase is this home on? is this hour a
weekday?). Doing them once here means no later file redoes them or gets them wrong.

Choices:
  - The home table is a complete grid (every home at every hour): a reading that never arrived keeps its
    row with empty values, so it counts as missing instead of vanishing.
  - Timestamps become real dates at one precision, so text (CSV) and dates (parquet) join.
  - Only the needed columns are kept: anything else (period, temperature...) is left behind.
  - Works for any number of hours: a whole year for learning, or recent history plus one new hour.
"""

import pandas as pd  # tables

from detector import settings  # phases and types of day
from detector.data_formats import (  # the columns each table needs or produces
    HOME_HOURS_COLUMNS,
    HOURLY_CONTEXT_COLUMNS,
    METER_READINGS_COLUMNS,
    PHASE_HOURS_COLUMNS,
    STREET_WIRING_COLUMNS,
    TRANSFORMER_READINGS_COLUMNS,
)


def to_timestamps(values):
    """Turns timestamps given as text or as dates into real dates, all at the same precision (microseconds).

    Two tables can only be joined on their timestamps if both hold the same kind of value:
    a CSV file gives text, a parquet file gives dates, and dates can be stored at different precisions.
    """
    return pd.to_datetime(values).dt.as_unit("us")


def day_types(hourly_context):
    """One row per hour: its timestamp and its type of day ("weekday" or "weekend_or_holiday")."""
    # Keep only the columns the detector needs: any other column (period, temperature...) is left behind.
    context = hourly_context[HOURLY_CONTEXT_COLUMNS].copy()
    context["timestamp"] = to_timestamps(context["timestamp"])
    # A weekend day or a public holiday is a day off; holidays join weekends.
    day_off = (context["is_weekend"] == 1) | (context["is_holiday"] == 1)
    weekday, weekend_or_holiday = settings.DAY_TYPES
    context["day_type"] = day_off.map({False: weekday, True: weekend_or_holiday})
    return context[["timestamp", "day_type"]]


def build_home_hours(meter_readings, street_wiring, hourly_context):
    """The home table: one row per home in the wiring and per hour in the context.

    Each row carries the home's phase, cable and contracted power, the hour's type of day, and the reading.
    A reading that never arrived still gets its row, with kwh, voltage_v and current_a empty.
    """
    # Every home paired with every hour (a "cross join"): the complete grid of rows the table must have.
    wiring = street_wiring[STREET_WIRING_COLUMNS]
    grid = wiring.merge(day_types(hourly_context), how="cross")

    # The readings, with their timestamps made comparable to the grid's.
    readings = meter_readings[METER_READINGS_COLUMNS].copy()
    readings["timestamp"] = to_timestamps(readings["timestamp"])

    # Attach each reading to its row of the grid (a "left join": every grid row is kept;
    # rows without a reading keep empty values instead of disappearing).
    table = grid.merge(readings, on=["home_id", "timestamp"], how="left")

    # A reading that never arrived is missing, not filled in by the utility: estimated = 0.
    table["estimated"] = table["estimated"].fillna(0).astype("int8")

    # Same order everywhere: home by home, then hour by hour; columns in the order of data_formats.py.
    table = table.sort_values(["home_id", "timestamp"], ignore_index=True)
    return table[HOME_HOURS_COLUMNS]


def build_phase_hours(home_hours, transformer_readings):
    """The phase table: one row per phase and hour, the transformer's measurement next to its homes' readings."""
    # The homes' side: for each hour and phase, add up what the homes reported.
    homes = (
        home_hours.groupby(["timestamp", "phase"])
        .agg(
            homes_kwh=("kwh", "sum"),  # sum of the kWh that arrived (empty values are skipped; 0 if none arrived)
            nb_homes_reporting=("kwh", "count"),  # number of homes whose kWh arrived (empty values aren't counted)
            nb_homes_on_phase=("home_id", "size"),  # number of homes on the phase, reading or not
            day_type=("day_type", "first"),  # the hour's type of day: the same for every home at that hour
        )
        .reset_index()
    )

    # The transformer's side: its file has one row per hour, with one column per phase (energy_kwh_A, ...).
    # Reshape it into one row per phase and hour, so it lines up with the homes' side.
    transformer = transformer_readings[TRANSFORMER_READINGS_COLUMNS].copy()
    transformer["timestamp"] = to_timestamps(transformer["timestamp"])
    per_phase = []
    for phase in settings.PHASES:
        one_phase = transformer[["timestamp", f"energy_kwh_{phase}", f"voltage_v_{phase}"]].rename(
            columns={f"energy_kwh_{phase}": "transformer_kwh", f"voltage_v_{phase}": "transformer_voltage_v"}
        )
        one_phase["phase"] = phase
        per_phase.append(one_phase)
    transformer_long = pd.concat(per_phase, ignore_index=True)

    # Put the two sides side by side. The homes' side decides which rows exist (a "left join"),
    # so an hour missing from the transformer's readings keeps its rows, with empty transformer values.
    table = homes.merge(transformer_long, on=["timestamp", "phase"], how="left")

    # Same order everywhere: hour by hour, then phase by phase; columns in the order of data_formats.py.
    table = table.sort_values(["timestamp", "phase"], ignore_index=True)
    return table[PHASE_HOURS_COLUMNS]
