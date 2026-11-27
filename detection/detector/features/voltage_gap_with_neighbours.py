"""The gap stage: each home's voltage gap with its neighbours at the new hour (plan step 4).

What it does: for each home, its voltage minus the median voltage of its neighbours (the other homes on its cable
and phase that the rules didn't untrust). One gap per home, in volts.

Why it exists: the feature voltage_gap_with_neighbours_vs_usual compares this gap with the home's usual gap. The
gap is computed once, when its hour is checked; the caller (check_hour) writes it into the home's row (column
voltage_gap_with_neighbours), so it goes into the memory with the row, because the gap's usual is the median of
past gaps.

Choices:
  - It runs after the rules, not in preparation: a neighbour the rules caught must be left out of the median (one
    0 V meter among three homes would give the other two huge fake gaps), and a home the rules caught gets no gap.
  - An estimated reading still counts as a neighbour: the rules don't untrust it, and one guessed voltage barely
    moves the median of 7 or 8 neighbours.
  - The median, not the average: one odd neighbour barely moves it.
  - It returns the gaps and leaves the home rows as they are: the caller writes the gaps in, as it will write each
    row's trust.
  - A plain loop over the homes: 75 homes an hour is small, and the loop reads in order.
"""

import numpy as np  # np.nan: an empty value
import pandas as pd  # tables


def voltage_gap_with_neighbours(new_hour_home, is_home_untrusted_by_rules):
    """Each home's voltage at the new hour minus the median voltage of its neighbours, in volts.

    new_hour_home: the home table's rows for the hour being checked (one row per home).
    is_home_untrusted_by_rules: one yes/no per home row, as trust.is_home_untrusted_by_rules returns it.
    Returns one gap per home row, in the same order. Empty when the rules untrusted the home, when it sent no
    voltage, or when none of its neighbours is usable (it's alone on its cable and phase, or they were all untrusted).
    """
    gaps = pd.Series(np.nan, index=new_hour_home.index)  # one empty gap per home, filled in below
    # One home at a time. iterrows gives each row's label (to write its gap in the right place) and the row's values.
    for row_label, home in new_hour_home.iterrows():
        if is_home_untrusted_by_rules.loc[row_label]:
            continue  # a reading the rules untrusted gets no gap: it stays empty
        # Its neighbours: same cable, same phase, not the home itself, and not untrusted by the rules.
        is_neighbour = (
            (new_hour_home["cable"] == home["cable"])
            & (new_hour_home["phase"] == home["phase"])
            & (new_hour_home["home_id"] != home["home_id"])
            & ~is_home_untrusted_by_rules  # "~" means "not"
        )
        # Their median voltage. median skips empty voltages, and is empty when no neighbour is left.
        neighbours_median_voltage = new_hour_home.loc[is_neighbour, "voltage_v"].median()
        gaps.loc[row_label] = home["voltage_v"] - neighbours_median_voltage
    return gaps
