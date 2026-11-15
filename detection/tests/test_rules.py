"""Tests for detector/rules/: the definition rules (plan step 2).

Each test cuts the made-up street of conftest.py the way the live system sees it at one hour: the new hour's
rows (one per home) and the memory (every earlier row). It plants one problem, runs a rule, and checks which
homes the rule answers yes for at the new hour.
"""

import pandas as pd  # tables
import pytest  # the test tool; @pytest.mark.parametrize runs one test once per value in a list

from detector.rules.meter_rules import find_impossible, find_missing, find_stuck  # the code being tested

# The three values of one reading (written out here, not imported from the code being tested).
READING = ["kwh", "voltage_v", "current_a"]


def friday(hour):
    """Friday 4 January 2019 at the given hour, e.g. friday(12) is 12:00."""
    return pd.Timestamp(2019, 1, 4, hour)


def at(table, home_id, hour):
    """True on the table's row for this home at this hour, False everywhere else: used to change that reading."""
    return (table["home_id"] == home_id) & (table["timestamp"] == hour)


def split_at(home_hours, hour):
    """Cuts the street as the live system sees it at that hour: the new hour's rows, and the memory (earlier rows)."""
    new_hour = home_hours[home_hours["timestamp"] == hour]
    memory = home_hours[home_hours["timestamp"] < hour]
    return new_hour, memory


def flagged_homes(new_hour, answers):
    """The homes a rule answered yes for, as a set, e.g. {"H3"}."""
    return set(new_hour.loc[answers, "home_id"])


def freeze(home_hours, home_id, last_real_hour, nb_copies):
    """Makes a meter stuck, as the dataset does: the nb_copies hours after last_real_hour repeat its whole reading."""
    real = home_hours.loc[at(home_hours, home_id, friday(last_real_hour)), READING].iloc[0]
    for hour in range(last_real_hour + 1, last_real_hour + 1 + nb_copies):
        for column in READING:
            home_hours.loc[at(home_hours, home_id, friday(hour)), column] = real[column]


# ============================================================================= an honest street
def test_honest_street_triggers_no_meter_rule(home_hours):
    """Every reading of the honest street is present, possible and moving: no meter rule may fire."""
    # Prepare: the street as the live system sees it at Friday 12:00.
    new_hour, memory = split_at(home_hours, friday(12))
    # Run: the three rules.
    answers_of_each_rule = [find_missing(new_hour), find_impossible(new_hour), find_stuck(new_hour, memory)]
    # Check: each rule gives one answer per home (6 homes), and every answer is no.
    for answers in answers_of_each_rule:
        assert len(answers) == 6
        assert not answers.any()


# ============================================================================= missing
def test_reading_that_never_arrived_is_missing(home_hours):
    """A home whose kWh, voltage and current are all empty is flagged missing."""
    # Prepare: empty H3's reading at 12:00, as when a meter sends nothing.
    home_hours.loc[at(home_hours, "H3", friday(12)), READING] = float("nan")
    new_hour, _memory = split_at(home_hours, friday(12))
    # Run and check: H3, and only H3, is flagged.
    assert flagged_homes(new_hour, find_missing(new_hour)) == {"H3"}


@pytest.mark.parametrize("empty_value", READING)  # runs 3 times: kwh, voltage_v, current_a
def test_reading_with_one_empty_value_is_missing(home_hours, empty_value):
    """A reading with any one of its three values empty can't be checked, so it counts as missing too."""
    # Prepare: empty only this value of H3's reading at 12:00; the other two are still there.
    home_hours.loc[at(home_hours, "H3", friday(12)), empty_value] = float("nan")
    new_hour, _memory = split_at(home_hours, friday(12))
    # Run and check: H3 is flagged.
    assert flagged_homes(new_hour, find_missing(new_hour)) == {"H3"}


# ============================================================================= impossible
def test_negative_energy_is_impossible(home_hours):
    """These homes have no solar panels, so energy only flows in: negative kWh is impossible."""
    # Prepare: H3 reports -0.4 kWh at 12:00.
    home_hours.loc[at(home_hours, "H3", friday(12)), "kwh"] = -0.4
    new_hour, _memory = split_at(home_hours, friday(12))
    # Run and check: H3, and only H3, is flagged.
    assert flagged_homes(new_hour, find_impossible(new_hour)) == {"H3"}


def test_energy_at_zero_volts_is_impossible(home_hours):
    """With no voltage, no energy can flow: kWh above 0 while the meter reads 0 V is impossible."""
    # Prepare: H3's voltage reads 0 V at 12:00, while its kWh stays as it was (0.462 kWh, above 0).
    home_hours.loc[at(home_hours, "H3", friday(12)), "voltage_v"] = 0.0
    new_hour, _memory = split_at(home_hours, friday(12))
    # Run and check: H3 is flagged.
    assert flagged_homes(new_hour, find_impossible(new_hour)) == {"H3"}


def test_no_energy_at_zero_volts_is_possible(home_hours):
    """0 kWh at 0 V is what a power cut looks like, not a broken meter: the impossible rule leaves it alone."""
    # Prepare: H3 reads 0 kWh, 0 V and 0 A at 12:00.
    home_hours.loc[at(home_hours, "H3", friday(12)), READING] = 0.0
    new_hour, _memory = split_at(home_hours, friday(12))
    # Run and check: nobody is flagged.
    assert flagged_homes(new_hour, find_impossible(new_hour)) == set()


def test_limit_is_one_and_a_half_times_each_homes_contract(home_hours):
    """Above 1.5 x the contracted power for a whole hour is impossible; a smaller overshoot is possible.

    Honest 2018 readings reached 1.29 x the contract, so 1.3 x must pass and 1.6 x must be caught.
    """
    # Prepare: H1 (3.3 kW contract) at 1.6 x = 5.28 kWh; H2 (4.4 kW contract) at 1.3 x = 5.72 kWh.
    home_hours.loc[at(home_hours, "H1", friday(12)), "kwh"] = 5.28
    home_hours.loc[at(home_hours, "H2", friday(12)), "kwh"] = 5.72
    new_hour, _memory = split_at(home_hours, friday(12))
    # Run and check: only H1 is flagged, although H2 used more energy: the limit follows each home's contract.
    assert flagged_homes(new_hour, find_impossible(new_hour)) == {"H1"}


def test_estimated_reading_is_never_impossible(home_hours):
    """An estimated value was filled in by the utility, not measured: it can't show a broken meter."""
    # Prepare: H3's reading at 12:00 is negative, but marked as estimated.
    is_h3_at_12 = at(home_hours, "H3", friday(12))
    home_hours.loc[is_h3_at_12, "kwh"] = -0.4
    home_hours.loc[is_h3_at_12, "estimated"] = 1
    new_hour, _memory = split_at(home_hours, friday(12))
    # Run and check: nobody is flagged.
    assert flagged_homes(new_hour, find_impossible(new_hour)) == set()


# ============================================================================= stuck
def test_reading_repeated_for_4_hours_is_stuck(home_hours):
    """4 identical whole readings in a row is more than honest meters ever repeat (3 in 2018): stuck."""
    # Prepare: H3 repeats its 09:00 reading at 10:00, 11:00 and 12:00 (4 identical readings).
    freeze(home_hours, "H3", last_real_hour=9, nb_copies=3)
    new_hour, memory = split_at(home_hours, friday(12))
    # Run and check: at 12:00, H3, and only H3, is stuck.
    assert flagged_homes(new_hour, find_stuck(new_hour, memory)) == {"H3"}


def test_reading_repeated_for_3_hours_is_not_stuck(home_hours):
    """3 identical readings in a row happens on honest meters: not stuck."""
    # Prepare: H3 repeats its 10:00 reading at 11:00 and 12:00; its 09:00 reading is different.
    freeze(home_hours, "H3", last_real_hour=10, nb_copies=2)
    new_hour, memory = split_at(home_hours, friday(12))
    # Run and check: nobody is stuck.
    assert flagged_homes(new_hour, find_stuck(new_hour, memory)) == set()


@pytest.mark.parametrize("moving_value", READING)  # runs 3 times: kwh, voltage_v, current_a
def test_one_moving_value_means_not_stuck(home_hours, moving_value):
    """Only a whole repeated reading is stuck: if any one of kWh, voltage or current keeps moving, it isn't.

    (An honest meter's energy can stay flat for hours, but its voltage keeps moving.)
    """
    # Prepare: H3 repeats its 09:00 reading until 12:00, except this one value, which keeps its honest values.
    honest = home_hours.copy()
    freeze(home_hours, "H3", last_real_hour=9, nb_copies=3)
    for hour in (10, 11, 12):
        is_h3_then = at(home_hours, "H3", friday(hour))
        home_hours.loc[is_h3_then, moving_value] = honest.loc[is_h3_then, moving_value]
    new_hour, memory = split_at(home_hours, friday(12))
    # Run and check: nobody is stuck, because the whole reading didn't repeat.
    assert flagged_homes(new_hour, find_stuck(new_hour, memory)) == set()


def test_estimated_hour_among_the_four_is_not_stuck(home_hours):
    """An estimated value isn't the meter's own reading, so it can't count as one of the 4 repeats."""
    # Prepare: H3 repeats its 09:00 reading until 12:00, but 10:00 is marked as estimated.
    freeze(home_hours, "H3", last_real_hour=9, nb_copies=3)
    home_hours.loc[at(home_hours, "H3", friday(10)), "estimated"] = 1
    new_hour, memory = split_at(home_hours, friday(12))
    # Run and check: nobody is stuck.
    assert flagged_homes(new_hour, find_stuck(new_hour, memory)) == set()


def test_hour_absent_from_the_memory_is_not_stuck(home_hours):
    """Without all 3 previous hours, the rule can't confirm 4 identical readings in a row: not stuck."""
    # Prepare: H3 repeats its 09:00 reading until 12:00, but the memory lost H3's 10:00 row (e.g. the app was down).
    freeze(home_hours, "H3", last_real_hour=9, nb_copies=3)
    new_hour, memory = split_at(home_hours, friday(12))
    memory_without_10 = memory[~at(memory, "H3", friday(10))]
    # Run and check: nobody is stuck.
    assert flagged_homes(new_hour, find_stuck(new_hour, memory_without_10)) == set()
