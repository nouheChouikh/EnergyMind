"""Tests for detector/rules/: the definition rules (plan step 2).

Each test plants one problem in the made-up street of conftest.py, then cuts it the way the live system sees it
at one hour: the new hour's rows and the memory (every earlier row). It runs a rule and checks which homes (or
phases) the rule answers yes for at the new hour.

Street tests follow one order: plant home problems in the home table, build the phase table from it (so the
homes' total and count include them), plant transformer problems in the phase table, then cut both tables.
"""

import pandas as pd  # tables
import pytest  # the test tool; @pytest.mark.parametrize runs one test once per value in a list

from detector.data_formats import RULE_FINDINGS_COLUMNS  # the findings table's expected columns
from detector.preparation.tables import build_home_hours, build_phase_hours  # builds the two tables
from detector.rules.findings import find_rule_findings  # the code being tested
from detector.rules.meter_rules import find_impossible, find_missing, find_stuck  # the code being tested
from detector.rules.street_rules import (  # the code being tested
    find_missing_all,
    find_outages,
    find_transformer_problems,
)


# =============================================================================
# SHARED HELPERS: used by both the meter tests and the street tests
# =============================================================================
# The three values of one home's reading (written out here, not imported from the code being tested).
READING = ["kwh", "voltage_v", "current_a"]


def friday(hour):
    """Friday 4 January 2019 at the given hour, e.g. friday(12) is 12:00."""
    return pd.Timestamp(2019, 1, 4, hour)


def at(table, home_id, hour):
    """True on the table's row for this home at this hour, False everywhere else: used to change that reading."""
    return (table["home_id"] == home_id) & (table["timestamp"] == hour)


def split_at(table, hour):
    """Cuts a table (home or phase table) as the live system sees it at that hour.

    Returns the new hour's rows, and the memory (every earlier row).
    """
    new_hour_rows = table[table["timestamp"] == hour]
    memory = table[table["timestamp"] < hour]
    return new_hour_rows, memory


# =============================================================================
# METER RULES (one home's reading): helpers
# =============================================================================
def flagged_homes(new_hour_home, answers):
    """The homes a rule answered yes for, as a set, e.g. {"H3"}."""
    return set(new_hour_home.loc[answers, "home_id"])


def freeze(home_hours, home_id, last_real_hour, nb_copies):
    """Makes a meter stuck, as the dataset does: the nb_copies hours after last_real_hour repeat its whole reading."""
    real = home_hours.loc[at(home_hours, home_id, friday(last_real_hour)), READING].iloc[0]
    for hour in range(last_real_hour + 1, last_real_hour + 1 + nb_copies):
        for column in READING:
            home_hours.loc[at(home_hours, home_id, friday(hour)), column] = real[column]


# =============================================================================
# METER RULES (one home's reading): tests
# =============================================================================
# ----------------------------------------------------------------------------- an honest street
def test_honest_street_triggers_no_meter_rule(home_hours):
    """Every reading of the honest street is present, possible and moving: no meter rule may fire."""
    # Prepare: the street as the live system sees it at Friday 12:00.
    new_hour_home, home_memory = split_at(home_hours, friday(12))
    # Run: the three rules.
    answers_of_each_rule = [
        find_missing(new_hour_home),
        find_impossible(new_hour_home),
        find_stuck(new_hour_home, home_memory),
    ]
    # Check: each rule gives one answer per home (6 homes), and every answer is no.
    for answers in answers_of_each_rule:
        assert len(answers) == 6
        assert not answers.any()


# ----------------------------------------------------------------------------- missing
def test_reading_that_never_arrived_is_missing(home_hours):
    """A home whose kWh, voltage and current are all empty is flagged missing."""
    # Prepare: empty H3's reading at 12:00, as when a meter sends nothing.
    home_hours.loc[at(home_hours, "H3", friday(12)), READING] = float("nan")
    new_hour_home, _home_memory = split_at(home_hours, friday(12))
    # Run and check: H3, and only H3, is flagged.
    assert flagged_homes(new_hour_home, find_missing(new_hour_home)) == {"H3"}


@pytest.mark.parametrize("empty_value", READING)  # runs 3 times: kwh, voltage_v, current_a
def test_reading_with_one_empty_value_is_missing(home_hours, empty_value):
    """A reading with any one of its three values empty can't be checked, so it counts as missing too."""
    # Prepare: empty only this value of H3's reading at 12:00; the other two are still there.
    home_hours.loc[at(home_hours, "H3", friday(12)), empty_value] = float("nan")
    new_hour_home, _home_memory = split_at(home_hours, friday(12))
    # Run and check: H3 is flagged.
    assert flagged_homes(new_hour_home, find_missing(new_hour_home)) == {"H3"}


# ----------------------------------------------------------------------------- impossible
def test_negative_energy_is_impossible(home_hours):
    """These homes have no solar panels, so energy only flows in: negative kWh is impossible."""
    # Prepare: H3 reports -0.4 kWh at 12:00.
    home_hours.loc[at(home_hours, "H3", friday(12)), "kwh"] = -0.4
    new_hour_home, _home_memory = split_at(home_hours, friday(12))
    # Run and check: H3, and only H3, is flagged.
    assert flagged_homes(new_hour_home, find_impossible(new_hour_home)) == {"H3"}


def test_energy_at_zero_volts_is_impossible(home_hours):
    """With no voltage, no energy can flow: kWh above 0 while the meter reads 0 V is impossible."""
    # Prepare: H3's voltage reads 0 V at 12:00, while its kWh stays as it was (0.462 kWh, above 0).
    home_hours.loc[at(home_hours, "H3", friday(12)), "voltage_v"] = 0.0
    new_hour_home, _home_memory = split_at(home_hours, friday(12))
    # Run and check: H3 is flagged.
    assert flagged_homes(new_hour_home, find_impossible(new_hour_home)) == {"H3"}


def test_no_energy_at_zero_volts_is_possible(home_hours):
    """0 kWh at 0 V is what a power cut looks like, not a broken meter: the impossible rule leaves it alone."""
    # Prepare: H3 reads 0 kWh, 0 V and 0 A at 12:00.
    home_hours.loc[at(home_hours, "H3", friday(12)), READING] = 0.0
    new_hour_home, _home_memory = split_at(home_hours, friday(12))
    # Run and check: nobody is flagged.
    assert flagged_homes(new_hour_home, find_impossible(new_hour_home)) == set()


def test_limit_is_one_and_a_half_times_each_homes_contract(home_hours):
    """Above 1.5 x the contracted power for a whole hour is impossible; a smaller overshoot is possible.

    Honest 2018 readings reached 1.29 x the contract, so 1.3 x must pass and 1.6 x must be caught.
    """
    # Prepare: H1 (3.3 kW contract) at 1.6 x = 5.28 kWh; H2 (4.4 kW contract) at 1.3 x = 5.72 kWh.
    home_hours.loc[at(home_hours, "H1", friday(12)), "kwh"] = 5.28
    home_hours.loc[at(home_hours, "H2", friday(12)), "kwh"] = 5.72
    new_hour_home, _home_memory = split_at(home_hours, friday(12))
    # Run and check: only H1 is flagged, although H2 used more energy: the limit follows each home's contract.
    assert flagged_homes(new_hour_home, find_impossible(new_hour_home)) == {"H1"}


def test_estimated_reading_is_never_impossible(home_hours):
    """An estimated value was filled in by the utility, not measured: it can't show a broken meter."""
    # Prepare: H3's reading at 12:00 is negative, but marked as estimated.
    is_h3_at_12 = at(home_hours, "H3", friday(12))
    home_hours.loc[is_h3_at_12, "kwh"] = -0.4
    home_hours.loc[is_h3_at_12, "estimated"] = 1
    new_hour_home, _home_memory = split_at(home_hours, friday(12))
    # Run and check: nobody is flagged.
    assert flagged_homes(new_hour_home, find_impossible(new_hour_home)) == set()


# ----------------------------------------------------------------------------- stuck
def test_reading_repeated_for_4_hours_is_stuck(home_hours):
    """4 identical whole readings in a row is more than honest meters ever repeat (3 in 2018): stuck."""
    # Prepare: H3 repeats its 09:00 reading at 10:00, 11:00 and 12:00 (4 identical readings).
    freeze(home_hours, "H3", last_real_hour=9, nb_copies=3)
    new_hour_home, home_memory = split_at(home_hours, friday(12))
    # Run and check: at 12:00, H3, and only H3, is stuck.
    assert flagged_homes(new_hour_home, find_stuck(new_hour_home, home_memory)) == {"H3"}


def test_reading_repeated_for_3_hours_is_not_stuck(home_hours):
    """3 identical readings in a row happens on honest meters: not stuck."""
    # Prepare: H3 repeats its 10:00 reading at 11:00 and 12:00; its 09:00 reading is different.
    freeze(home_hours, "H3", last_real_hour=10, nb_copies=2)
    new_hour_home, home_memory = split_at(home_hours, friday(12))
    # Run and check: nobody is stuck.
    assert flagged_homes(new_hour_home, find_stuck(new_hour_home, home_memory)) == set()


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
    new_hour_home, home_memory = split_at(home_hours, friday(12))
    # Run and check: nobody is stuck, because the whole reading didn't repeat.
    assert flagged_homes(new_hour_home, find_stuck(new_hour_home, home_memory)) == set()


def test_estimated_hour_among_the_four_is_not_stuck(home_hours):
    """An estimated value isn't the meter's own reading, so it can't count as one of the 4 repeats."""
    # Prepare: H3 repeats its 09:00 reading until 12:00, but 10:00 is marked as estimated.
    freeze(home_hours, "H3", last_real_hour=9, nb_copies=3)
    home_hours.loc[at(home_hours, "H3", friday(10)), "estimated"] = 1
    new_hour_home, home_memory = split_at(home_hours, friday(12))
    # Run and check: nobody is stuck.
    assert flagged_homes(new_hour_home, find_stuck(new_hour_home, home_memory)) == set()


def test_hour_absent_from_the_memory_is_not_stuck(home_hours):
    """Without all 3 previous hours, the rule can't confirm 4 identical readings in a row: not stuck."""
    # Prepare: H3 repeats its 09:00 reading until 12:00, but the memory lost H3's 10:00 row (e.g. the app was down).
    freeze(home_hours, "H3", last_real_hour=9, nb_copies=3)
    new_hour_home, home_memory = split_at(home_hours, friday(12))
    memory_without_10 = home_memory[~at(home_memory, "H3", friday(10))]
    # Run and check: nobody is stuck.
    assert flagged_homes(new_hour_home, find_stuck(new_hour_home, memory_without_10)) == set()


def test_zero_volt_readings_never_count_as_stuck(home_hours):
    """A stuck meter freezes on a real reading; 0 V means no power reached the meter, so it is never "stuck"."""
    # Prepare: H3 reads 0 kWh, 0 V, 0 A from 09:00 to 12:00 (4 identical readings), with power on its phase.
    for hour in (9, 10, 11, 12):
        home_hours.loc[at(home_hours, "H3", friday(hour)), READING] = 0.0
    new_hour_home, home_memory = split_at(home_hours, friday(12))
    # Run and check: nobody is stuck (0 V is an outage or a home without supply, not a frozen reading).
    assert flagged_homes(new_hour_home, find_stuck(new_hour_home, home_memory)) == set()


# =============================================================================
# STREET RULES (the transformer and whole phases): helpers
# =============================================================================
# The transformer's reading on one phase, in the phase table (it measures energy and voltage, not current).
TRANSFORMER = ["transformer_kwh", "transformer_voltage_v"]


def on_phase(table, phase, hour):
    """True on this phase's rows at this hour: one row in the phase table, the phase's homes in the home table."""
    return (table["phase"] == phase) & (table["timestamp"] == hour)


def cut_homes(home_hours, phase, hour):
    """The home side of a power cut, as the dataset makes it: every home on the phase reads 0 kWh, 0 V, 0 A."""
    home_hours.loc[on_phase(home_hours, phase, hour), READING] = 0.0


def freeze_transformer(phases, phase, last_real_hour, nb_copies):
    """Makes the transformer's meter stuck on one phase: the nb_copies hours after last_real_hour repeat its reading."""
    real = phases.loc[on_phase(phases, phase, friday(last_real_hour)), TRANSFORMER].iloc[0]
    for hour in range(last_real_hour + 1, last_real_hour + 1 + nb_copies):
        for column in TRANSFORMER:
            phases.loc[on_phase(phases, phase, friday(hour)), column] = real[column]


def street_at(home_hours, phases, hour):
    """Cuts both tables as the live system sees them at that hour.

    Returns the new hour's home rows, the new hour's phase rows, and the phase memory (earlier phase rows).
    """
    new_hour_home, _home_memory = split_at(home_hours, hour)
    new_hour_phase, phase_memory = split_at(phases, hour)
    return new_hour_home, new_hour_phase, phase_memory


def flagged_phases(new_hour_phase, answers):
    """The phases a rule answered yes for, as a set, e.g. {"B"}."""
    return set(new_hour_phase.loc[answers, "phase"])


def street_with_ten_homes_on_b(meter_readings, street_wiring, hourly_context):
    """The made-up street's home table with 8 more homes on phase B (copies of H3, named B1 to B8).

    Phase B then has 10 homes: with only 2, one odd meter would already be half of the phase.
    """
    copies = [f"B{number}" for number in range(1, 9)]
    h3_wiring = street_wiring[street_wiring["home_id"] == "H3"]
    h3_readings = meter_readings[meter_readings["home_id"] == "H3"]
    wiring = pd.concat([street_wiring] + [h3_wiring.assign(home_id=name) for name in copies], ignore_index=True)
    readings = pd.concat([meter_readings] + [h3_readings.assign(home_id=name) for name in copies], ignore_index=True)
    return build_home_hours(readings, wiring, hourly_context)


# =============================================================================
# STREET RULES (the transformer and whole phases): tests
# =============================================================================
# ----------------------------------------------------------------------------- an honest street
def test_honest_street_triggers_no_street_rule(home_hours, transformer_readings):
    """The transformer and every home measure normally: no street rule may fire."""
    # Prepare: the street as the live system sees it at Friday 12:00.
    phases = build_phase_hours(home_hours, transformer_readings)
    new_hour_home, new_hour_phase, phase_memory = street_at(home_hours, phases, friday(12))
    # Run: the three street rules.
    outages = find_outages(new_hour_home, new_hour_phase)
    transformer_problems = find_transformer_problems(new_hour_home, new_hour_phase, phase_memory)
    # Check: one answer per phase (3 phases), every answer no; and the street isn't silent.
    for answers in (outages, transformer_problems):
        assert len(answers) == 3
        assert not answers.any()
    assert not find_missing_all(new_hour_home, new_hour_phase)


# ----------------------------------------------------------------------------- outages
def test_power_cut_on_one_phase_is_an_outage(home_hours, transformer_readings):
    """Phase B's transformer and homes all read 0: an outage on phase B only, and no transformer problem."""
    # Prepare: phase B loses power at 12:00. Its homes read 0...
    cut_homes(home_hours, "B", friday(12))
    phases = build_phase_hours(home_hours, transformer_readings)
    # ...and so does the transformer's phase B.
    phases.loc[on_phase(phases, "B", friday(12)), TRANSFORMER] = 0.0
    new_hour_home, new_hour_phase, phase_memory = street_at(home_hours, phases, friday(12))
    # Run and check: an outage on B only; the transformer isn't blamed, since its homes agree with it.
    assert flagged_phases(new_hour_phase, find_outages(new_hour_home, new_hour_phase)) == {"B"}
    transformer_problems = find_transformer_problems(new_hour_home, new_hour_phase, phase_memory)
    assert flagged_phases(new_hour_phase, transformer_problems) == set()


def test_power_cut_with_silent_meters_is_an_outage(home_hours, transformer_readings):
    """A meter without power may send nothing: silent homes count as homes without voltage."""
    # Prepare: at 12:00 phase B's homes send nothing...
    home_hours.loc[on_phase(home_hours, "B", friday(12)), READING] = float("nan")
    phases = build_phase_hours(home_hours, transformer_readings)
    # ...and the transformer's phase B reads 0 kWh, 0 V.
    phases.loc[on_phase(phases, "B", friday(12)), TRANSFORMER] = 0.0
    new_hour_home, new_hour_phase, _phase_memory = street_at(home_hours, phases, friday(12))
    # Run and check: an outage on B.
    assert flagged_phases(new_hour_phase, find_outages(new_hour_home, new_hour_phase)) == {"B"}


def test_homes_at_zero_under_a_measuring_transformer_is_no_outage(home_hours, transformer_readings):
    """An outage needs both sides: if the transformer still delivers energy and voltage, the phase has power."""
    # Prepare: at 12:00 phase B's homes read 0, but the transformer's phase B measures normally.
    cut_homes(home_hours, "B", friday(12))
    phases = build_phase_hours(home_hours, transformer_readings)
    new_hour_home, new_hour_phase, _phase_memory = street_at(home_hours, phases, friday(12))
    # Run and check: no outage.
    assert flagged_phases(new_hour_phase, find_outages(new_hour_home, new_hour_phase)) == set()


def test_power_cut_on_every_phase_is_an_outage_on_each(home_hours, transformer_readings):
    """A street-wide power cut is an outage on all 3 phases (findings.py turns it into one street finding)."""
    # Prepare: at 12:00 every home reads 0...
    for phase in ("A", "B", "C"):
        cut_homes(home_hours, phase, friday(12))
    phases = build_phase_hours(home_hours, transformer_readings)
    # ...and the transformer reads 0 kWh, 0 V on every phase.
    phases.loc[phases["timestamp"] == friday(12), TRANSFORMER] = 0.0
    new_hour_home, new_hour_phase, _phase_memory = street_at(home_hours, phases, friday(12))
    # Run and check: an outage on A, B and C.
    assert flagged_phases(new_hour_phase, find_outages(new_hour_home, new_hour_phase)) == {"A", "B", "C"}


def test_one_odd_meter_cannot_block_an_outage(meter_readings, street_wiring, hourly_context, transformer_readings):
    """In a real outage, one meter in ten still showing voltage (e.g. stuck on its last value) can't block it."""
    # Prepare: phase B has 10 homes. At 12:00 they all read 0, except H3, which still shows a normal reading...
    home_hours = street_with_ten_homes_on_b(meter_readings, street_wiring, hourly_context)
    cut_homes(home_hours, "B", friday(12))
    home_hours.loc[at(home_hours, "H3", friday(12)), READING] = [0.4, 229.3, 1.8]
    phases = build_phase_hours(home_hours, transformer_readings)
    # ...and the transformer's phase B reads 0 kWh, 0 V.
    phases.loc[on_phase(phases, "B", friday(12)), TRANSFORMER] = 0.0
    new_hour_home, new_hour_phase, phase_memory = street_at(home_hours, phases, friday(12))
    # Run and check: 9 homes in 10 without voltage is nearly all: an outage on B, and no transformer problem.
    assert flagged_phases(new_hour_phase, find_outages(new_hour_home, new_hour_phase)) == {"B"}
    transformer_problems = find_transformer_problems(new_hour_home, new_hour_phase, phase_memory)
    assert flagged_phases(new_hour_phase, transformer_problems) == set()


# ----------------------------------------------------------------------------- transformer problems
def test_transformer_at_zero_while_its_homes_have_power(home_hours, transformer_readings):
    """The transformer's phase B reads 0 kWh, 0 V, but its homes show voltage: the transformer's meter is wrong."""
    # Prepare: at 12:00 the transformer's phase B reads 0 kWh, 0 V; its homes measure normally.
    phases = build_phase_hours(home_hours, transformer_readings)
    phases.loc[on_phase(phases, "B", friday(12)), TRANSFORMER] = 0.0
    new_hour_home, new_hour_phase, phase_memory = street_at(home_hours, phases, friday(12))
    # Run and check: a transformer problem on B, and no outage.
    transformer_problems = find_transformer_problems(new_hour_home, new_hour_phase, phase_memory)
    assert flagged_phases(new_hour_phase, transformer_problems) == {"B"}
    assert flagged_phases(new_hour_phase, find_outages(new_hour_home, new_hour_phase)) == set()


def test_nine_homes_in_ten_with_power_is_enough(meter_readings, street_wiring, hourly_context, transformer_readings):
    """One home in ten reading 0 V can't excuse a transformer at 0: 9 in 10 with voltage is nearly all."""
    # Prepare: phase B has 10 homes; at 12:00 only H3 reads 0, the other 9 measure normally...
    home_hours = street_with_ten_homes_on_b(meter_readings, street_wiring, hourly_context)
    home_hours.loc[at(home_hours, "H3", friday(12)), READING] = 0.0
    phases = build_phase_hours(home_hours, transformer_readings)
    # ...while the transformer's phase B reads 0 kWh, 0 V.
    phases.loc[on_phase(phases, "B", friday(12)), TRANSFORMER] = 0.0
    new_hour_home, new_hour_phase, phase_memory = street_at(home_hours, phases, friday(12))
    # Run and check: a transformer problem on B, and no outage.
    transformer_problems = find_transformer_problems(new_hour_home, new_hour_phase, phase_memory)
    assert flagged_phases(new_hour_phase, transformer_problems) == {"B"}
    assert flagged_phases(new_hour_phase, find_outages(new_hour_home, new_hour_phase)) == set()


@pytest.mark.parametrize("empty_value", TRANSFORMER)  # runs 2 times: transformer_kwh, transformer_voltage_v
def test_transformer_reading_with_an_empty_value(home_hours, transformer_readings, empty_value):
    """A transformer reading with a hole in it can't be used: a transformer problem."""
    # Prepare: at 12:00 one value of the transformer's phase B reading is empty.
    phases = build_phase_hours(home_hours, transformer_readings)
    phases.loc[on_phase(phases, "B", friday(12)), empty_value] = float("nan")
    new_hour_home, new_hour_phase, phase_memory = street_at(home_hours, phases, friday(12))
    # Run and check: a transformer problem on B only.
    transformer_problems = find_transformer_problems(new_hour_home, new_hour_phase, phase_memory)
    assert flagged_phases(new_hour_phase, transformer_problems) == {"B"}


def test_negative_transformer_energy(home_hours, transformer_readings):
    """The transformer only delivers energy to the street: negative energy is a transformer problem."""
    # Prepare: at 12:00 the transformer's phase B reports -1.0 kWh.
    phases = build_phase_hours(home_hours, transformer_readings)
    phases.loc[on_phase(phases, "B", friday(12)), "transformer_kwh"] = -1.0
    new_hour_home, new_hour_phase, phase_memory = street_at(home_hours, phases, friday(12))
    # Run and check: a transformer problem on B.
    transformer_problems = find_transformer_problems(new_hour_home, new_hour_phase, phase_memory)
    assert flagged_phases(new_hour_phase, transformer_problems) == {"B"}


def test_transformer_energy_at_zero_volts(home_hours, transformer_readings):
    """Energy can't flow without voltage, at the transformer as in a home: a transformer problem."""
    # Prepare: at 12:00 the transformer's phase B reads 0 V while its energy stays as it was (above 0).
    phases = build_phase_hours(home_hours, transformer_readings)
    phases.loc[on_phase(phases, "B", friday(12)), "transformer_voltage_v"] = 0.0
    new_hour_home, new_hour_phase, phase_memory = street_at(home_hours, phases, friday(12))
    # Run and check: a transformer problem on B.
    transformer_problems = find_transformer_problems(new_hour_home, new_hour_phase, phase_memory)
    assert flagged_phases(new_hour_phase, transformer_problems) == {"B"}


def test_transformer_reading_repeated_for_4_hours(home_hours, transformer_readings):
    """A real voltage always moves: the same transformer reading 4 hours in a row means its meter is stuck."""
    # Prepare: the transformer's phase B repeats its 09:00 reading at 10:00, 11:00 and 12:00.
    phases = build_phase_hours(home_hours, transformer_readings)
    freeze_transformer(phases, "B", last_real_hour=9, nb_copies=3)
    new_hour_home, new_hour_phase, phase_memory = street_at(home_hours, phases, friday(12))
    # Run and check: at 12:00, a transformer problem on B.
    transformer_problems = find_transformer_problems(new_hour_home, new_hour_phase, phase_memory)
    assert flagged_phases(new_hour_phase, transformer_problems) == {"B"}


def test_transformer_reading_repeated_for_3_hours(home_hours, transformer_readings):
    """3 identical readings in a row isn't enough to call the transformer's meter stuck (same limit as meters)."""
    # Prepare: the transformer's phase B repeats its 10:00 reading at 11:00 and 12:00; 09:00 is different.
    phases = build_phase_hours(home_hours, transformer_readings)
    freeze_transformer(phases, "B", last_real_hour=10, nb_copies=2)
    new_hour_home, new_hour_phase, phase_memory = street_at(home_hours, phases, friday(12))
    # Run and check: no transformer problem.
    transformer_problems = find_transformer_problems(new_hour_home, new_hour_phase, phase_memory)
    assert flagged_phases(new_hour_phase, transformer_problems) == set()


@pytest.mark.parametrize("moving_value", TRANSFORMER)  # runs 2 times: transformer_kwh, transformer_voltage_v
def test_one_moving_transformer_value_means_not_stuck(home_hours, transformer_readings, moving_value):
    """Only a whole repeated transformer reading is stuck: if its energy or its voltage keeps moving, it isn't."""
    # Prepare: phase B repeats its 09:00 reading until 12:00, except this one value, which keeps its honest values.
    phases = build_phase_hours(home_hours, transformer_readings)
    honest = phases.copy()
    freeze_transformer(phases, "B", last_real_hour=9, nb_copies=3)
    for hour in (10, 11, 12):
        is_b_then = on_phase(phases, "B", friday(hour))
        phases.loc[is_b_then, moving_value] = honest.loc[is_b_then, moving_value]
    new_hour_home, new_hour_phase, phase_memory = street_at(home_hours, phases, friday(12))
    # Run and check: no transformer problem.
    transformer_problems = find_transformer_problems(new_hour_home, new_hour_phase, phase_memory)
    assert flagged_phases(new_hour_phase, transformer_problems) == set()


def test_repeated_zero_during_a_real_outage_is_not_stuck(home_hours, transformer_readings):
    """During a 4-hour outage the transformer repeats 0 kWh, 0 V: that's the outage, not a stuck meter."""
    # Prepare: phase B loses power from 09:00 to 12:00. Its homes read 0...
    for hour in (9, 10, 11, 12):
        cut_homes(home_hours, "B", friday(hour))
    phases = build_phase_hours(home_hours, transformer_readings)
    # ...and so does the transformer's phase B, 4 hours in a row.
    for hour in (9, 10, 11, 12):
        phases.loc[on_phase(phases, "B", friday(hour)), TRANSFORMER] = 0.0
    new_hour_home, new_hour_phase, phase_memory = street_at(home_hours, phases, friday(12))
    # Run and check: at 12:00, an outage on B and no transformer problem.
    assert flagged_phases(new_hour_phase, find_outages(new_hour_home, new_hour_phase)) == {"B"}
    transformer_problems = find_transformer_problems(new_hour_home, new_hour_phase, phase_memory)
    assert flagged_phases(new_hour_phase, transformer_problems) == set()


def test_transformer_at_zero_for_hours_while_homes_have_power(home_hours, transformer_readings):
    """A transformer stuck at 0 kWh, 0 V while its homes have power is caught at every hour, from the first."""
    # Prepare: the transformer's phase B reads 0 kWh, 0 V from 09:00 to 12:00; its homes measure normally.
    phases = build_phase_hours(home_hours, transformer_readings)
    for hour in (9, 10, 11, 12):
        phases.loc[on_phase(phases, "B", friday(hour)), TRANSFORMER] = 0.0
    # Run and check, hour by hour as live: a transformer problem on B at 09:00, 10:00, 11:00 and 12:00.
    for hour in (9, 10, 11, 12):
        new_hour_home, new_hour_phase, phase_memory = street_at(home_hours, phases, friday(hour))
        transformer_problems = find_transformer_problems(new_hour_home, new_hour_phase, phase_memory)
        assert flagged_phases(new_hour_phase, transformer_problems) == {"B"}


def test_transformer_broken_on_two_phases(home_hours, transformer_readings):
    """Each phase is judged on its own: broken on B and C gives a problem on exactly those two."""
    # Prepare: at 12:00 phase B reports negative energy and phase C's voltage is empty.
    phases = build_phase_hours(home_hours, transformer_readings)
    phases.loc[on_phase(phases, "B", friday(12)), "transformer_kwh"] = -1.0
    phases.loc[on_phase(phases, "C", friday(12)), "transformer_voltage_v"] = float("nan")
    new_hour_home, new_hour_phase, phase_memory = street_at(home_hours, phases, friday(12))
    # Run and check: problems on B and C, not on A.
    transformer_problems = find_transformer_problems(new_hour_home, new_hour_phase, phase_memory)
    assert flagged_phases(new_hour_phase, transformer_problems) == {"B", "C"}


def test_homes_that_disagree_give_no_street_verdict(home_hours, transformer_readings):
    """Transformer at 0 kWh, 0 V with 1 home of 2 showing voltage: the evidence contradicts itself, so no rule fires."""
    # Prepare: at 12:00 H3 reads 0 while H4 (also on phase B) measures normally...
    home_hours.loc[at(home_hours, "H3", friday(12)), READING] = 0.0
    phases = build_phase_hours(home_hours, transformer_readings)
    # ...and the transformer's phase B reads 0 kWh, 0 V.
    phases.loc[on_phase(phases, "B", friday(12)), TRANSFORMER] = 0.0
    new_hour_home, new_hour_phase, phase_memory = street_at(home_hours, phases, friday(12))
    # Run and check: neither an outage nor a transformer problem (the forest and an operator will look at it).
    assert flagged_phases(new_hour_phase, find_outages(new_hour_home, new_hour_phase)) == set()
    transformer_problems = find_transformer_problems(new_hour_home, new_hour_phase, phase_memory)
    assert flagged_phases(new_hour_phase, transformer_problems) == set()


# ----------------------------------------------------------------------------- all meters missing
def test_every_meter_silent_while_the_transformer_measures(home_hours, transformer_readings):
    """No meter sends anything, but the transformer still measures: the data link failed, not the power."""
    # Prepare: at 12:00 every home's reading is empty; the transformer measures normally.
    home_hours.loc[home_hours["timestamp"] == friday(12), READING] = float("nan")
    phases = build_phase_hours(home_hours, transformer_readings)
    new_hour_home, new_hour_phase, _phase_memory = street_at(home_hours, phases, friday(12))
    # Run and check: all meters missing.
    assert find_missing_all(new_hour_home, new_hour_phase)


def test_five_meters_in_six_silent_is_not_nearly_all(home_hours, transformer_readings):
    """5 of 6 meters silent is 83%, below the 90% of "nearly all": not a data link failure."""
    # Prepare: at 12:00 every home except H6 sends nothing.
    is_12 = home_hours["timestamp"] == friday(12)
    home_hours.loc[is_12 & (home_hours["home_id"] != "H6"), READING] = float("nan")
    phases = build_phase_hours(home_hours, transformer_readings)
    new_hour_home, new_hour_phase, _phase_memory = street_at(home_hours, phases, friday(12))
    # Run and check: not all meters missing.
    assert not find_missing_all(new_hour_home, new_hour_phase)


def test_every_meter_silent_in_a_street_power_cut_is_an_outage(home_hours, transformer_readings):
    """Meters without power may send nothing: with the transformer at 0 everywhere, the silence is an outage."""
    # Prepare: at 12:00 every home sends nothing...
    home_hours.loc[home_hours["timestamp"] == friday(12), READING] = float("nan")
    phases = build_phase_hours(home_hours, transformer_readings)
    # ...and the transformer reads 0 kWh, 0 V on every phase.
    phases.loc[phases["timestamp"] == friday(12), TRANSFORMER] = 0.0
    new_hour_home, new_hour_phase, _phase_memory = street_at(home_hours, phases, friday(12))
    # Run and check: not a data link failure, but an outage on all 3 phases.
    assert not find_missing_all(new_hour_home, new_hour_phase)
    assert flagged_phases(new_hour_phase, find_outages(new_hour_home, new_hour_phase)) == {"A", "B", "C"}


# =============================================================================
# FINDINGS (all rules together): helpers
# =============================================================================
def findings_at(home_hours, phases, hour):
    """Runs every rule at that hour, as the live system would, and returns the findings table."""
    new_hour_home, home_memory = split_at(home_hours, hour)
    new_hour_phase, phase_memory = split_at(phases, hour)
    return find_rule_findings(new_hour_home, new_hour_phase, home_memory, phase_memory)


def rows(findings):
    """The findings as a set of (scope, phase, home_id, verdict), e.g. {("phase", "B", None, "outage_phase")}.

    Empty cells become None (pandas may store them as None or as NaN, depending on the column type).
    Fails the test if two findings are identical, since a set would hide the duplicate.
    """
    def none_if_empty(value):
        return None if pd.isna(value) else value

    as_tuples = [
        tuple(none_if_empty(value) for value in row)
        for row in findings[["scope", "phase", "home_id", "verdict"]].itertuples(index=False, name=None)
    ]
    assert len(set(as_tuples)) == len(as_tuples)
    return set(as_tuples)


# =============================================================================
# FINDINGS (all rules together): tests
# =============================================================================
# ----------------------------------------------------------------------------- an honest street
def test_honest_street_gives_an_empty_findings_table(home_hours, transformer_readings):
    """No rule fires on the honest street: the findings table is empty, but has its 5 columns."""
    # Prepare: the honest street's phase table.
    phases = build_phase_hours(home_hours, transformer_readings)
    # Run: every rule at Friday 12:00.
    findings = findings_at(home_hours, phases, friday(12))
    # Check: no row, and exactly the columns listed in data_formats.py.
    assert len(findings) == 0
    assert list(findings.columns) == RULE_FINDINGS_COLUMNS


# ----------------------------------------------------------------------------- street findings
def test_phase_outage_covers_its_homes(home_hours, transformer_readings):
    """Phase B out: one outage_phase row on B, dated at the new hour, and no row for its homes H3 and H4.

    H4 sends nothing (a meter without power may not send), so the missing rule fires on it: only the outage
    covering its homes keeps H4 off the list.
    """
    # Prepare: phase B loses power at 12:00. H3 reads 0, H4 sends nothing...
    home_hours.loc[at(home_hours, "H3", friday(12)), READING] = 0.0
    home_hours.loc[at(home_hours, "H4", friday(12)), READING] = float("nan")
    phases = build_phase_hours(home_hours, transformer_readings)
    # ...and the transformer's phase B reads 0 kWh, 0 V.
    phases.loc[on_phase(phases, "B", friday(12)), TRANSFORMER] = 0.0
    # Run.
    findings = findings_at(home_hours, phases, friday(12))
    # Check: exactly one row, the outage on B, at 12:00.
    assert rows(findings) == {("phase", "B", None, "outage_phase")}
    assert (findings["timestamp"] == friday(12)).all()


def test_two_phases_out_give_two_phase_rows(home_hours, transformer_readings):
    """Phases A and B out, C fine: one outage_phase row for each phase that's out."""
    # Prepare: phases A and B lose power at 12:00 (their homes and the transformer read 0).
    for phase in ("A", "B"):
        cut_homes(home_hours, phase, friday(12))
    phases = build_phase_hours(home_hours, transformer_readings)
    for phase in ("A", "B"):
        phases.loc[on_phase(phases, phase, friday(12)), TRANSFORMER] = 0.0
    # Run.
    findings = findings_at(home_hours, phases, friday(12))
    # Check: two rows, one per phase out.
    assert rows(findings) == {("phase", "A", None, "outage_phase"), ("phase", "B", None, "outage_phase")}


def test_every_phase_out_gives_one_street_row(home_hours, transformer_readings):
    """All 3 phases out: one outage_street row, not 3 phase rows."""
    # Prepare: the whole street loses power at 12:00 (every home and every transformer phase read 0).
    home_hours.loc[home_hours["timestamp"] == friday(12), READING] = 0.0
    phases = build_phase_hours(home_hours, transformer_readings)
    phases.loc[phases["timestamp"] == friday(12), TRANSFORMER] = 0.0
    # Run.
    findings = findings_at(home_hours, phases, friday(12))
    # Check: exactly one row, for the whole street.
    assert rows(findings) == {("street", None, None, "outage_street")}


def test_long_street_outage_never_turns_into_stuck_meters(home_hours, transformer_readings):
    """A 6-hour street outage, checked hour by hour as live: one street row every hour, never a stuck meter."""
    # Prepare: the whole street loses power from 09:00 to 14:00 (6 hours of 0 everywhere).
    outage_hours = [friday(hour) for hour in range(9, 15)]
    home_hours.loc[home_hours["timestamp"].isin(outage_hours), READING] = 0.0
    phases = build_phase_hours(home_hours, transformer_readings)
    phases.loc[phases["timestamp"].isin(outage_hours), TRANSFORMER] = 0.0
    # Run and check, hour by hour: only the street outage, every hour.
    for hour in outage_hours:
        assert rows(findings_at(home_hours, phases, hour)) == {("street", None, None, "outage_street")}


def test_every_meter_silent_gives_one_street_row(home_hours, transformer_readings):
    """Every meter silent while the transformer measures: one missing_data_all_meters row, not 6 home rows."""
    # Prepare: at 12:00 no meter sends anything; the transformer measures normally.
    home_hours.loc[home_hours["timestamp"] == friday(12), READING] = float("nan")
    phases = build_phase_hours(home_hours, transformer_readings)
    # Run.
    findings = findings_at(home_hours, phases, friday(12))
    # Check: exactly one row, for the whole street.
    assert rows(findings) == {("street", None, None, "missing_data_all_meters")}


def test_all_meters_missing_covers_even_the_meter_that_sent(meter_readings, street_wiring, hourly_context,
                                                            transformer_readings):
    """During a data link failure, the one meter that did send gets no row: its reading came through a failing
    collection system, so it can't be trusted."""
    # Prepare: a 14-home street (6 homes need all 6 silent to reach 90%). At 12:00, 13 of 14 meters (93%)
    # send nothing, and the one that did, H5, sends an impossible negative reading.
    home_hours = street_with_ten_homes_on_b(meter_readings, street_wiring, hourly_context)
    is_12 = home_hours["timestamp"] == friday(12)
    home_hours.loc[is_12 & (home_hours["home_id"] != "H5"), READING] = float("nan")
    home_hours.loc[at(home_hours, "H5", friday(12)), "kwh"] = -0.4
    phases = build_phase_hours(home_hours, transformer_readings)
    # Run.
    findings = findings_at(home_hours, phases, friday(12))
    # Check: only the street row; H5 gets no row of its own.
    assert rows(findings) == {("street", None, None, "missing_data_all_meters")}


def test_transformer_problem_gives_one_transformer_row(home_hours, transformer_readings):
    """Transformer broken on B: one reference_meter_problem row on B; its homes are fine, so they get no row."""
    # Prepare: at 12:00 the transformer's phase B reports negative energy; every home measures normally.
    phases = build_phase_hours(home_hours, transformer_readings)
    phases.loc[on_phase(phases, "B", friday(12)), "transformer_kwh"] = -1.0
    # Run.
    findings = findings_at(home_hours, phases, friday(12))
    # Check: exactly one row, the transformer's meter on B.
    assert rows(findings) == {("transformer", "B", None, "reference_meter_problem")}


# ----------------------------------------------------------------------------- home findings
def test_home_problems_give_one_row_each_with_their_phase(home_hours, transformer_readings):
    """H1 impossible, H3 silent and H5 stuck at 12:00: three home rows, each with its home's phase."""
    # Prepare: H1 (phase A) reports -0.4 kWh, H3 (phase B) sends nothing, H5 (phase C) repeats its 09:00 reading.
    home_hours.loc[at(home_hours, "H1", friday(12)), "kwh"] = -0.4
    home_hours.loc[at(home_hours, "H3", friday(12)), READING] = float("nan")
    freeze(home_hours, "H5", last_real_hour=9, nb_copies=3)
    phases = build_phase_hours(home_hours, transformer_readings)
    # Run.
    findings = findings_at(home_hours, phases, friday(12))
    # Check: one row per home, with its phase and its verdict.
    assert rows(findings) == {
        ("home", "A", "H1", "impossible_value"),
        ("home", "B", "H3", "missing_data_one_meter"),
        ("home", "C", "H5", "stuck_meter"),
    }


def test_meter_frozen_on_an_impossible_value_is_reported_as_impossible(home_hours, transformer_readings):
    """H3 repeats a negative reading for 4 hours: both impossible and stuck, but one verdict, impossible_value."""
    # Prepare: H3 reads -0.4 kWh, 229.3 V, 1.8 A at 09:00, 10:00, 11:00 and 12:00.
    for hour in (9, 10, 11, 12):
        home_hours.loc[at(home_hours, "H3", friday(hour)), READING] = [-0.4, 229.3, 1.8]
    phases = build_phase_hours(home_hours, transformer_readings)
    # Run.
    findings = findings_at(home_hours, phases, friday(12))
    # Check: exactly one row for H3, impossible (the stronger statement).
    assert rows(findings) == {("home", "B", "H3", "impossible_value")}


def test_phase_outage_and_a_home_problem_elsewhere_both_appear(home_hours, transformer_readings):
    """Phase B out and H5 (on phase C) impossible: the outage covers only phase B's homes, so both rows appear."""
    # Prepare: phase B loses power at 12:00 (its homes and the transformer read 0); H5 reports -0.4 kWh.
    cut_homes(home_hours, "B", friday(12))
    home_hours.loc[at(home_hours, "H5", friday(12)), "kwh"] = -0.4
    phases = build_phase_hours(home_hours, transformer_readings)
    phases.loc[on_phase(phases, "B", friday(12)), TRANSFORMER] = 0.0
    # Run.
    findings = findings_at(home_hours, phases, friday(12))
    # Check: the outage on B, and H5's impossible reading.
    assert rows(findings) == {("phase", "B", None, "outage_phase"), ("home", "C", "H5", "impossible_value")}
