"""Street rules: a power cut, a silent street, or a broken transformer meter at the new hour? (plan step 2)

What it does: three rules judge the new hour by comparing the transformer with its homes:
  - outage: a transformer phase reads 0 kWh and 0 V, and nearly all of its homes show no voltage
    (0 V, or nothing sent: a meter without power may not send);
  - transformer problem: the transformer's own meter (the reference meter) is wrong on a phase: an empty value,
    negative energy, energy at 0 V, 0 kWh and 0 V while nearly all of its homes show voltage, or the same
    reading 4 hours in a row;
  - all meters missing: nearly all of the street's meters sent nothing, while the transformer shows no
    street-wide power cut (the data link failed, not the power).

Why it exists: a power cut, a broken data link or a broken transformer meter makes many readings look wrong at
once. Naming the one street problem stops it from turning into dozens of false house problems.

Choices:
  - Built for live use, like the meter rules: each rule judges the new hour only. The transformer-stuck check
    takes the transformer's 3 previous hours from the memory.
  - "Nearly all" (90%), not "every home": one odd meter (e.g. stuck on its last value) can't block a clear
    verdict. When the homes disagree among themselves (10% to 90% show voltage), no rule fires: the evidence
    contradicts itself, so the forest and an operator look at it.
  - Outage and transformer problem never both fire on the same phase and hour, so their order doesn't matter.
  - The transformer-stuck check skips readings at 0 V: the other checks judge those at once, and a real
    outage's repeated 0 kWh, 0 V must never be called stuck.
  - The homes' side is read from the home rows (voltages, missing readings), with the meter rule's own
    definition of a missing reading.
"""

import pandas as pd  # tables

from detector import settings  # "nearly all" (90%) and 4 identical readings
from detector.rules.meter_rules import find_missing  # the one definition of a missing reading

# The transformer's reading on one phase, in the phase table (it measures energy and voltage, not current).
TRANSFORMER_READING = ["transformer_kwh", "transformer_voltage_v"]


# =============================================================================
# SHARED HELPER: the homes' side of each phase
# =============================================================================
def share_of_homes(new_hour_home, new_hour_phase, condition):
    """For each row of new_hour_phase, the share (0 to 1) of that phase's homes for which condition is True.

    condition: one True/False per row of new_hour_home. A phase without homes gets an empty share.
    """
    # One share per phase, e.g. A: 0.0, B: 1.0, C: 0.5 (the average of True/False is the share of True).
    share_by_phase = condition.groupby(new_hour_home["phase"]).mean()
    # Lined up with new_hour_phase's rows.
    return new_hour_phase["phase"].map(share_by_phase)


# =============================================================================
# OUTAGES
# =============================================================================
def find_outages(new_hour_home, new_hour_phase):
    """True for each phase that lost power at the new hour: the transformer and nearly all of its homes agree.

    new_hour_home: the home table's rows for the hour being checked (one row per home).
    new_hour_phase: the phase table's rows for that hour (one row per phase).
    Returns one True/False per row of new_hour_phase, in the same order.
    """
    # The transformer's side: this phase delivered no energy and has no voltage.
    transformer_at_zero = (new_hour_phase["transformer_kwh"] == 0) & (new_hour_phase["transformer_voltage_v"] == 0)
    # The homes' side: nearly all homes on the phase show no voltage.
    # (An empty voltage fails "> 0", so a meter that sent nothing counts as a home without voltage.)
    no_voltage = ~(new_hour_home["voltage_v"] > 0)
    homes_without_power = share_of_homes(new_hour_home, new_hour_phase, no_voltage) >= settings.NEARLY_ALL_SHARE
    return transformer_at_zero & homes_without_power


# =============================================================================
# TRANSFORMER PROBLEMS (its own meter, the reference meter, is wrong)
# =============================================================================
def find_transformer_problems(new_hour_home, new_hour_phase, phase_memory):
    """True for each phase where the transformer's own meter is wrong at the new hour.

    new_hour_home: the home table's rows for the hour being checked (one row per home).
    new_hour_phase: the phase table's rows for that hour (one row per phase).
    phase_memory: the phase table's rows for earlier hours, as kept by the operator app. It may hold more hours
        and more columns: only each phase's 3 previous transformer readings are used.
    Returns one True/False per row of new_hour_phase, in the same order.
    """
    kwh = new_hour_phase["transformer_kwh"]
    voltage = new_hour_phase["transformer_voltage_v"]
    # The reading has a hole in it.
    empty = kwh.isna() | voltage.isna()
    # Negative energy: the transformer only delivers energy to the street.
    negative = kwh < 0
    # Energy without voltage: impossible, as for a home.
    energy_at_zero_volts = (voltage == 0) & (kwh > 0)
    # 0 kWh and 0 V while nearly all homes on the phase show voltage: the homes have power, so the transformer's
    # meter is the one that's wrong.
    shows_voltage = new_hour_home["voltage_v"] > 0
    homes_with_power = share_of_homes(new_hour_home, new_hour_phase, shows_voltage) >= settings.NEARLY_ALL_SHARE
    zero_while_homes_have_power = (kwh == 0) & (voltage == 0) & homes_with_power
    # The same reading 4 hours in a row.
    stuck = transformer_stuck(new_hour_phase, phase_memory)
    return empty | negative | energy_at_zero_volts | zero_while_homes_have_power | stuck


def transformer_stuck(new_hour_phase, phase_memory):
    """True for each phase whose transformer reading (kWh and V) is identical to its 3 readings before it."""
    # Every earlier transformer reading, findable by phase and hour.
    columns = ["phase", "timestamp", *TRANSFORMER_READING]
    earlier_readings = phase_memory[columns].set_index(["phase", "timestamp"])

    # Readings at 0 V are left out: the other checks judge them at once (outage, energy at 0 V, 0 kWh and 0 V
    # while the homes have power), and a real outage's repeated 0 kWh, 0 V must never be called stuck.
    stuck = new_hour_phase["transformer_voltage_v"] > 0
    for hours_back in range(1, settings.STUCK_MIN_HOURS):  # 1, 2 and 3 hours before the new hour
        # Each phase's reading hours_back hours before. An hour absent from the memory comes back empty.
        wanted = pd.MultiIndex.from_arrays(
            [new_hour_phase["phase"], new_hour_phase["timestamp"] - pd.Timedelta(hours=hours_back)]
        )
        earlier = earlier_readings.reindex(wanted)
        earlier.index = new_hour_phase.index  # line each earlier reading up with its phase's row in new_hour_phase
        # Identical in kWh and voltage. An empty value never equals anything, so it breaks the run.
        identical = (earlier[TRANSFORMER_READING] == new_hour_phase[TRANSFORMER_READING]).all(axis=1)
        stuck = stuck & identical
    return stuck


# =============================================================================
# ALL METERS MISSING (the data link failed)
# =============================================================================
def find_missing_all(new_hour_home, new_hour_phase):
    """True if nearly all of the street's meters sent nothing at the new hour, while the power didn't go.

    new_hour_home: the home table's rows for the hour being checked (one row per home).
    new_hour_phase: the phase table's rows for that hour (one row per phase).
    Returns a single True/False for the hour.
    """
    # The share of the street's meters whose reading is missing (same definition as the meter rule).
    share_missing = find_missing(new_hour_home).mean()
    # If the transformer also reads 0 kWh and 0 V on every phase, the silence is a street-wide power cut (the
    # outage rule names it), not a data link failure.
    transformer_at_zero = (new_hour_phase["transformer_kwh"] == 0) & (new_hour_phase["transformer_voltage_v"] == 0)
    street_cut = transformer_at_zero.all()
    return bool(share_missing >= settings.NEARLY_ALL_SHARE and not street_cut)
