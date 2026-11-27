"""The usual: what a home's or a phase's value normally is at this hour (plan step 4).

What it does: for the new hour, gives each home's usual consumption, each home's usual voltage gap with its
neighbours, and each phase's usual transformer voltage:
  - normally, the median of the same hour, on the same type of day, over the last 28 calendar days, using the
    memory's trusted rows only, when there are at least 3 of them;
  - otherwise, the last usual remembered for that hour and type of day (the last usuals table);
  - otherwise (none ever computed, e.g. the first days of the data), empty.
Every new median is also remembered: it overwrites the old value in the last usuals table.

Why it exists: three of the five features compare the new hour with "usual". Only trusted rows feed it, so problems
don't distort it; the remembered usual keeps a home scored during a long theft, when too few trusted values remain.

Choices:
  - The median, not the average: a few odd hours barely move it.
  - Same hour and same type of day: a Tuesday at 18:00 is compared with weekdays at 18:00 only.
  - 28 calendar days: keeps the comparison in the current season. Near the start of the data there are fewer days
    before the new hour: the values that exist count, so a home's first usual comes on its 4th weekday (or 4th
    weekend day) at that hour; before that it has none and is unchecked.
  - Built for live use: the usual comes only from the memory's past rows; the new hour is what gets compared.
  - The last usuals live in their own table because the memory only keeps about 28 days.
  - One shared function does the work for all three usuals; each one only says which value it reads and under
    which name it's remembered.
"""

import pandas as pd  # tables

from detector import settings  # the 28 days, the minimum of 3 values, the trust and usual names


# =============================================================================
# THE THREE USUALS
# =============================================================================
def consumption_usuals(new_hour_home, home_memory, last_usuals):
    """Each home's usual consumption (kWh) at the new hour, and the last usuals updated.

    new_hour_home: the home table's rows for the hour being checked (one row per home).
    home_memory: the memory's past home rows, with their trust (HOME_MEMORY_COLUMNS).
    last_usuals: the last usuals table (LAST_USUALS_COLUMNS), kept by the app; empty at the start.
    Returns (usuals, updated_last_usuals): one usual per home row, in the same order (empty for a home with none
    yet), and the last usuals with this hour's new medians written in.
    """
    return usuals_at_new_hour(
        new_hour_rows=new_hour_home,
        past_rows=home_memory,
        home_or_phase_column="home_id",
        value_column="kwh",
        last_usuals=last_usuals,
        usual_of=settings.USUAL_OF_CONSUMPTION,
    )


def voltage_gap_with_neighbours_usuals(new_hour_home, home_memory, last_usuals):
    """Each home's usual voltage gap with its neighbours (V) at the new hour, and the last usuals updated.

    new_hour_home: the home table's rows for the hour being checked (one row per home).
    home_memory: the memory's past home rows, with their gap and their trust (HOME_MEMORY_COLUMNS).
    last_usuals: the last usuals table (LAST_USUALS_COLUMNS), kept by the app; empty at the start.
    Returns (usuals, updated_last_usuals): one usual gap per home row, in the same order (empty for a home with none
    yet), and the last usuals with this hour's new medians written in.
    """
    return usuals_at_new_hour(
        new_hour_rows=new_hour_home,
        past_rows=home_memory,
        home_or_phase_column="home_id",
        value_column="voltage_gap_with_neighbours",
        last_usuals=last_usuals,
        usual_of=settings.USUAL_OF_VOLTAGE_GAP_WITH_NEIGHBOURS,
    )


def transformer_voltage_usuals(new_hour_phase, phase_memory, last_usuals):
    """Each phase's usual transformer voltage at the new hour, and the last usuals updated.

    new_hour_phase: the phase table's rows for the hour being checked (one row per phase).
    phase_memory: the memory's past phase rows, with their trust (PHASE_MEMORY_COLUMNS).
    last_usuals: the last usuals table (LAST_USUALS_COLUMNS), kept by the app; empty at the start.
    Returns (usuals, updated_last_usuals): one usual per phase row, in the same order (empty for a phase with none
    yet), and the last usuals with this hour's new medians written in.
    """
    return usuals_at_new_hour(
        new_hour_rows=new_hour_phase,
        past_rows=phase_memory,
        home_or_phase_column="phase",
        value_column="transformer_voltage_v",
        last_usuals=last_usuals,
        usual_of=settings.USUAL_OF_TRANSFORMER_VOLTAGE,
    )


# =============================================================================
# THE WORK, SHARED BY EVERY USUAL
# =============================================================================
def usuals_at_new_hour(new_hour_rows, past_rows, home_or_phase_column, value_column, last_usuals, usual_of):
    """The usual of one value for each row of the new hour, and the last usuals updated.

    new_hour_rows: the new hour's rows (one per home, or one per phase).
    past_rows: the memory's rows for earlier hours, with their trust.
    home_or_phase_column: the column saying whose value it is: "home_id" or "phase".
    value_column: the value whose usual is wanted, e.g. "kwh".
    last_usuals: the last usuals table (LAST_USUALS_COLUMNS).
    usual_of: this usual's name in the last usuals table, e.g. settings.USUAL_OF_CONSUMPTION.
    """
    new_hour = new_hour_rows["timestamp"].iloc[0]  # the hour being checked (the same on every row)
    day_type = new_hour_rows["day_type"].iloc[0]  # its type of day (the same on every row)

    # 1. The past rows that count: same hour of day, same type of day, within the last 28 days, trusted.
    #    (Near the start of the data there are fewer than 28 days before the new hour: the rows that exist count.)
    window_start = new_hour - pd.Timedelta(days=settings.USUAL_WINDOW_DAYS)
    is_same_hour_of_day = past_rows["timestamp"].dt.hour == new_hour.hour
    is_same_day_type = past_rows["day_type"] == day_type
    is_in_window = (past_rows["timestamp"] >= window_start) & (past_rows["timestamp"] < new_hour)
    is_trusted = past_rows["trust"] == settings.TRUST_TRUSTED
    counted_rows = past_rows[is_same_hour_of_day & is_same_day_type & is_in_window & is_trusted]

    # 2. Each home's (or phase's) median, kept only when it comes from at least 3 values.
    values_by_home_or_phase = counted_rows.groupby(home_or_phase_column)[value_column]
    medians = values_by_home_or_phase.median()
    nb_values = values_by_home_or_phase.count()  # empty values aren't counted
    new_medians = medians[nb_values >= settings.USUAL_MIN_VALUES]

    # 3. The usuals already remembered for this usual (e.g. consumption), this hour of day and this type of day.
    is_same_usual_hour_and_day_type = (
        (last_usuals["usual_of"] == usual_of)
        & (last_usuals["hour_of_day"] == new_hour.hour)
        & (last_usuals["day_type"] == day_type)
    )
    remembered_rows = last_usuals[is_same_usual_hour_and_day_type]
    remembered_usuals = remembered_rows.set_index("home_or_phase")["usual"].astype(float)  # findable by home

    # 4. Each row's usual: its new median; if it has none, its remembered usual; if nothing is remembered, empty.
    homes_or_phases = new_hour_rows[home_or_phase_column]  # each row's home (or phase)
    usuals = homes_or_phases.map(new_medians)  # empty where there were fewer than 3 values
    usuals = usuals.fillna(homes_or_phases.map(remembered_usuals))  # fill those with the remembered usual, if any
    usuals = usuals.astype(float)  # as decimal numbers

    # 5. Remember the new medians: each one overwrites the old value for the same home (or phase).
    is_overwritten = is_same_usual_hour_and_day_type & last_usuals["home_or_phase"].isin(new_medians.index)
    kept_rows = last_usuals[~is_overwritten]  # every remembered usual except the ones being overwritten
    new_rows = pd.DataFrame({
        "usual_of": usual_of,
        "home_or_phase": new_medians.index,
        "hour_of_day": new_hour.hour,
        "day_type": day_type,
        "usual": new_medians.to_numpy(),
    })
    updated_last_usuals = pd.concat([kept_rows, new_rows], ignore_index=True)  # the kept usuals, then the new ones
    return usuals, updated_last_usuals
