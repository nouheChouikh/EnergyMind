"""Meter rules: is one meter's reading missing, impossible or stuck at the new hour?

What it does: each rule receives the home table's rows for the new hour (one row per home) and answers yes or
no for each home:
  - missing: kWh, voltage or current is empty;
  - impossible: negative kWh, kWh above 0 at 0 V, or kWh above 1.5 x the home's contracted power;
  - stuck: the new reading is identical to the meter's 3 readings before it (taken from the memory).

Why it exists: these problems are definitions, so a rule states them exactly. A reading caught here is not
scored by the forest and is marked untrusted, so it never feeds the usual.

Choices:
  - Built for live use: a rule judges the new hour only, once, and never goes back to an earlier hour.
    Learning replays 2018 hour by hour through these same functions.
  - Estimated readings (filled in by the utility, not measured) are never impossible or stuck.
  - 1.5 x the contract: honest 2018 readings reached 1.29 x. Far above it, the number itself must be wrong,
    since the meter cuts the power when a home draws too much.
  - Stuck needs the whole reading (kWh, V and I): an honest meter's energy can stay flat, its voltage moves.
    Honest meters repeat for up to 3 readings (2018), so the alarm rings at the 4th: 2 hours after the freeze.
  - These rules know nothing about power cuts: findings.py runs the street rules first and keeps one
    verdict per home-hour.
"""

import pandas as pd  # tables

from detector import settings  # the limits: 1.5 x the contract, 4 identical readings

# The three values of one reading. The meter sends them together, so it is missing or stuck as a whole.
READING = ["kwh", "voltage_v", "current_a"]


def find_missing(new_hour_home):
    """True for each home whose reading at the new hour has an empty value (kWh, voltage or current).

    new_hour_home: the home table's rows for the hour being checked (one row per home).
    Returns one True/False per row of new_hour_home, in the same order.
    """
    # One empty value is enough: a reading with a hole in it can't be checked.
    return new_hour_home[READING].isna().any(axis=1)


def find_impossible(new_hour_home):
    """True for each home whose measured reading at the new hour can't physically happen.

    new_hour_home: the home table's rows for the hour being checked (one row per home).
    Returns one True/False per row of new_hour_home, in the same order.
    """
    kwh = new_hour_home["kwh"]
    # Negative energy: these homes have no solar panels, so energy only flows in.
    negative = kwh < 0
    # Energy without voltage: with no voltage, no current flows, so no energy can be used.
    energy_at_zero_volts = (new_hour_home["voltage_v"] == 0) & (kwh > 0)
    # Far too much energy: the kWh used in one hour is the average kW over that hour, and the meter cuts the
    # power when a home draws more than its contract, so it can't be far above the contracted kW.
    above_contract = kwh > settings.IMPOSSIBLE_KWH_FACTOR * new_hour_home["contracted_kw"]
    # Only measured readings count: an estimated value was filled in by the utility, not by the meter.
    # (An empty value fails every comparison above, so a missing reading is never also impossible.)
    measured = new_hour_home["estimated"] == 0
    return measured & (negative | energy_at_zero_volts | above_contract)


def find_stuck(new_hour_home, home_memory):
    """True for each home whose reading at the new hour is identical to its 3 readings before it.

    new_hour_home: the home table's rows for the hour being checked (one row per home).
    home_memory: the home table's rows for earlier hours, as kept by the operator app. It may hold more hours
        and more columns (like trust): only each meter's 3 previous readings are used.
    Returns one True/False per row of new_hour_home, in the same order.
    """
    # Every recent reading, findable by home and hour.
    columns = ["home_id", "timestamp", *READING, "estimated"]
    recent = pd.concat([home_memory[columns], new_hour_home[columns]]).set_index(["home_id", "timestamp"])

    # Start with every home stuck, then clear each home whose last readings break the run.
    stuck = pd.Series(True, index=new_hour_home.index)
    for hours_back in range(settings.STUCK_MIN_HOURS):  # 0 (the new hour itself), then 1, 2 and 3 hours before
        # Each home's reading hours_back hours before the new hour. An hour absent from the memory comes back empty.
        wanted = pd.MultiIndex.from_arrays(
            [new_hour_home["home_id"], new_hour_home["timestamp"] - pd.Timedelta(hours=hours_back)]
        )
        earlier = recent.reindex(wanted)
        earlier.index = new_hour_home.index  # line each earlier reading up with its home's row in new_hour_home
        # That reading must be measured, and identical to the new one in kWh, voltage and current.
        # An empty value never equals anything, so an empty or absent reading breaks the run.
        measured = earlier["estimated"] == 0
        identical = (earlier[READING] == new_hour_home[READING]).all(axis=1)
        stuck = stuck & measured & identical
    return stuck
