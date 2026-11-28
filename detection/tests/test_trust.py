"""Tests for detector/trust.py: which home rows and phase rows may feed "usual" (plan step 4, the trust rule).

What it does: checks trust.py's three parts:
  - the rules' answers, on the made-up street of conftest.py: which homes and which phases the rules make untrusted
    at the new hour, for each kind of finding and for several findings at once;
  - which home rows the forest can score, and the trust states during learning, on small hand-written tables.

Why it exists: trust decides what the usuals learn from. If a caught reading slipped through as trusted, it would
feed a usual, and every later hour would be compared with a polluted "normal".

Choices:
  - Findings are planted by hand (with finding() below), so each test holds exactly the finding it needs. Trust reads
    only the findings, so the street's readings stay honest.
  - finding() and findings_table() are copied from test_features.py (the losses tests use them too): about ten lines,
    simpler than a shared helper file.
  - The scorable rows and the trust states read only a few columns, so their tests write small tables by hand: each
    row holds exactly the case it tests.
  - The expected trust states are written out ("trusted", "untrusted", "unchecked") rather than imported, so a
    change to the agreed names is noticed.
  - Test names start with what they test (test_untrusted_by_rules_..., test_is_home_row_scorable_...,
    test_home_trust_states_during_learning_..., test_phase_trust_states_during_learning_...), so the test report
    reads in order.
  - Names say what a value holds: a yes/no for each row starts with is_; a set of names is named after what's in it
    (untrusted_homes). An answer keeps the full name of the function that gives it (is_home_untrusted_by_rules): the
    function is called through its file (trust.is_home_untrusted_by_rules), so the two names never clash.
"""

import numpy as np  # np.nan: an empty value
import pandas as pd  # tables
import pytest  # the test tool; @pytest.mark.parametrize runs one test once per case in a list

from detector import settings  # the scope and verdict names a finding is written with
from detector import trust  # the code being tested, called through its file (e.g. trust.is_home_untrusted_by_rules)
from detector.data_formats import (  # the tables' columns
    FOREST_TABLE_COLUMNS,
    HOME_HOURS_COLUMNS,
    PHASE_HOURS_COLUMNS,
    RULE_FINDINGS_COLUMNS,
)
from detector.preparation.tables import build_phase_hours  # builds the phase table from the made-up street


# =============================================================================
# HELPERS
# =============================================================================
# The tests use the made-up street of conftest.py at Friday 12:00 (H1 and H2 on phase A, H3 and H4 on B, H5 and H6
# on C), with findings planted by hand. Trust reads only the findings, so the readings themselves stay honest.
TRUST_HOUR = pd.Timestamp(2019, 1, 4, 12)  # the new hour every test in this file checks: Friday 12:00
ALL_HOMES = {"H1", "H2", "H3", "H4", "H5", "H6"}  # every home of the made-up street
ALL_PHASES = {"A", "B", "C"}


def finding(hour, scope, verdict, phase=None, home_id=None):
    """One rule finding, as a row of the findings table.

    Written out here rather than produced by the rules, so a test can plant exactly the finding it needs.
    phase: for a home, the home's phase; for the transformer, the broken phase; empty (None) for the whole street.
    """
    return {"timestamp": hour, "scope": scope, "phase": phase, "home_id": home_id, "verdict": verdict}


def findings_table(findings):
    """A findings table holding the findings in this list; findings_table([]) means no rule fired."""
    return pd.DataFrame(findings, columns=RULE_FINDINGS_COLUMNS)


def street_at_trust_hour(home_hours, transformer_readings):
    """The made-up street at TRUST_HOUR, as the live system sees a new hour: its home rows and its phase rows."""
    phase_hours = build_phase_hours(home_hours, transformer_readings)  # the phase table, built the real way
    new_hour_home = home_hours[home_hours["timestamp"] == TRUST_HOUR]
    new_hour_phase = phase_hours[phase_hours["timestamp"] == TRUST_HOUR]
    return new_hour_home, new_hour_phase


def names_untrusted_by_rules(home_hours, transformer_readings, findings):
    """The homes and the phases the rules make untrusted at TRUST_HOUR, given these findings.

    Returns two sets of names, e.g. ({"H3"}, set()) when only H3 is untrusted.
    """
    new_hour_home, new_hour_phase = street_at_trust_hour(home_hours, transformer_readings)
    rule_findings = findings_table(findings)
    # The code being tested answers True or False for each row.
    is_home_untrusted_by_rules = trust.is_home_untrusted_by_rules(new_hour_home, rule_findings)
    is_phase_untrusted_by_rules = trust.is_phase_untrusted_by_rules(new_hour_phase, rule_findings)
    # Keep the names of the rows answered True.
    untrusted_homes = set(new_hour_home.loc[is_home_untrusted_by_rules, "home_id"])
    untrusted_phases = set(new_hour_phase.loc[is_phase_untrusted_by_rules, "phase"])
    return untrusted_homes, untrusted_phases


# =============================================================================
# WHAT THE RULES MAKE UNTRUSTED AT THE NEW HOUR
# =============================================================================
# ----------------------------------------------------------------------------- no finding
def test_untrusted_by_rules_nothing_without_findings(home_hours, transformer_readings):
    """When no rule fired, the rules untrust nothing: one answer per row, and every answer is False."""
    # Prepare: the street at Friday 12:00, and an empty findings table.
    new_hour_home, new_hour_phase = street_at_trust_hour(home_hours, transformer_readings)
    no_findings = findings_table([])
    # Run: both functions.
    is_home_untrusted_by_rules = trust.is_home_untrusted_by_rules(new_hour_home, no_findings)
    is_phase_untrusted_by_rules = trust.is_phase_untrusted_by_rules(new_hour_phase, no_findings)
    # Check: 6 answers for the 6 homes and 3 for the 3 phases, every one False.
    assert list(is_home_untrusted_by_rules) == [False] * 6
    assert list(is_phase_untrusted_by_rules) == [False] * 3


# ----------------------------------------------------------------------------- one finding at a time
def test_untrusted_by_rules_home_finding_untrusts_only_that_home(home_hours, transformer_readings):
    """A home problem untrusts only that home: its phase row stays, since the transformer's reading is fine."""
    # Prepare: H3, a home on phase B, is caught stuck.
    stuck = finding(TRUST_HOUR, settings.SCOPE_HOME, settings.VERDICT_STUCK, phase="B", home_id="H3")
    # Run and check: only H3 is untrusted; no phase row is.
    assert names_untrusted_by_rules(home_hours, transformer_readings, [stuck]) == ({"H3"}, set())


def test_untrusted_by_rules_transformer_problem_untrusts_only_its_phase_row(home_hours, transformer_readings):
    """A broken transformer reading untrusts its phase row only: the homes' own readings are fine."""
    # Prepare: the transformer's reading on phase A is broken.
    broken = finding(TRUST_HOUR, settings.SCOPE_TRANSFORMER, settings.VERDICT_REFERENCE_METER, phase="A")
    # Run and check: no home is untrusted; only phase A's row is.
    assert names_untrusted_by_rules(home_hours, transformer_readings, [broken]) == (set(), {"A"})


def test_untrusted_by_rules_phase_outage_untrusts_its_homes_and_phase_row(home_hours, transformer_readings):
    """An outage untrusts everything it touches: every home on the phase, and the phase row.

    The covered homes have no finding row of their own (findings.py writes one row for the whole outage), so this
    also checks that the outage's phase is read.
    """
    # Prepare: phase C is out.
    outage = finding(TRUST_HOUR, settings.SCOPE_PHASE, settings.VERDICT_OUTAGE_PHASE, phase="C")
    # Run and check: H5 and H6 (the homes on C) and phase C's row are untrusted.
    assert names_untrusted_by_rules(home_hours, transformer_readings, [outage]) == ({"H5", "H6"}, {"C"})


def test_untrusted_by_rules_street_outage_untrusts_everything(home_hours, transformer_readings):
    """A street outage touches the whole street: every home and every phase row is untrusted."""
    # Prepare: all three phases are out (one street finding, with no phase).
    outage = finding(TRUST_HOUR, settings.SCOPE_STREET, settings.VERDICT_OUTAGE_STREET)
    # Run and check: every home and every phase row is untrusted.
    assert names_untrusted_by_rules(home_hours, transformer_readings, [outage]) == (ALL_HOMES, ALL_PHASES)


def test_untrusted_by_rules_all_meters_missing_untrusts_homes_not_phases(home_hours, transformer_readings):
    """All meters missing untrusts every home, but no phase row: the data link failed, not the transformer.

    Readings that slip through a failing collection system can't be trusted, but the transformer's reading arrived.
    """
    # Prepare: the street's meters went silent (one street finding, with no phase).
    silent = finding(TRUST_HOUR, settings.SCOPE_STREET, settings.VERDICT_MISSING_ALL)
    # Run and check: every home is untrusted; no phase row is.
    assert names_untrusted_by_rules(home_hours, transformer_readings, [silent]) == (ALL_HOMES, set())


# ----------------------------------------------------------------------------- several findings at once
def test_untrusted_by_rules_findings_add_up(home_hours, transformer_readings):
    """Several findings in the same hour add up: each one untrusts what it touches."""
    # Prepare: phase C is out and H3 (on phase B) is stuck, written in the order findings.py writes them.
    outage = finding(TRUST_HOUR, settings.SCOPE_PHASE, settings.VERDICT_OUTAGE_PHASE, phase="C")
    stuck = finding(TRUST_HOUR, settings.SCOPE_HOME, settings.VERDICT_STUCK, phase="B", home_id="H3")
    # Run and check: H3 for its own finding, H5 and H6 for the outage; only phase C's row.
    assert names_untrusted_by_rules(home_hours, transformer_readings, [outage, stuck]) == ({"H3", "H5", "H6"}, {"C"})


# =============================================================================
# WHICH HOME ROWS THE FOREST CAN SCORE, AND THE TRUST STATES: helpers
# =============================================================================
# These tests write their tables by hand, at TRUST_HOUR. pd.DataFrame(rows, columns=...) makes a table holding the
# rows given, with exactly the columns named.
def home_row(home_id, phase, estimated=0):
    """One home row of the new hour. Only the home, its phase and its estimated flag (1 = the utility filled the
    value in itself) matter here; the other columns hold ordinary values."""
    return {
        "timestamp": TRUST_HOUR,
        "home_id": home_id,
        "phase": phase,
        "cable": 1,
        "contracted_kw": 3.3,
        "day_type": "weekday",
        "kwh": 0.5,
        "voltage_v": 230.0,
        "current_a": 1.0,
        "estimated": estimated,
    }


def phase_row(phase):
    """One phase row of the new hour. Only the phase matters here; the other columns hold ordinary values."""
    return {
        "timestamp": TRUST_HOUR,
        "phase": phase,
        "day_type": "weekday",
        "transformer_kwh": 40.0,
        "transformer_voltage_v": 231.0,
        "homes_kwh": 39.0,
        "nb_homes_reporting": 25,
        "nb_homes_on_phase": 25,
    }


def forest_table_row(home_id, consumption_vs_usual=1.0, voltage_gap_with_neighbours_vs_usual=0.0,
                     phase_unexplained_energy=0.0, transformer_voltage_vs_usual=0.0):
    """One row of the forest table at the new hour: every feature filled in with an ordinary value, unless given
    (np.nan: empty)."""
    return {
        "timestamp": TRUST_HOUR,
        "home_id": home_id,
        "consumption_vs_usual": consumption_vs_usual,
        "voltage_gap_with_neighbours_vs_usual": voltage_gap_with_neighbours_vs_usual,
        "phase_unexplained_energy": phase_unexplained_energy,
        "transformer_voltage_vs_usual": transformer_voltage_vs_usual,
        "hour": TRUST_HOUR.hour,  # 12
    }


# =============================================================================
# WHICH HOME ROWS THE FOREST CAN SCORE (is_home_row_scorable)
# =============================================================================
def test_is_home_row_scorable_leaves_out_caught_and_estimated_readings():
    """The forest can score a row the rules didn't untrust, that was measured, with all five features. A caught
    reading is broken, and an estimated one is the utility's guess: neither is scored."""
    # Prepare: the rules caught H1, H2's reading is estimated, H3 is neither; all three have every feature filled in.
    new_hour_home = pd.DataFrame(
        [home_row("H1", "A"), home_row("H2", "A", estimated=1), home_row("H3", "A")], columns=HOME_HOURS_COLUMNS
    )
    forest_table = pd.DataFrame(
        [forest_table_row("H1"), forest_table_row("H2"), forest_table_row("H3")], columns=FOREST_TABLE_COLUMNS
    )
    is_home_untrusted_by_rules = pd.Series([True, False, False])  # the rules untrusted H1 only
    # Run.
    is_home_row_scorable = trust.is_home_row_scorable(new_hour_home, is_home_untrusted_by_rules, forest_table)
    # Check: no for H1 (caught), no for H2 (estimated), yes for H3.
    assert list(is_home_row_scorable) == [False, False, True]


# Each case empties one feature. The forest scores a whole row at once, so with any one feature empty there's
# nothing complete to score.
@pytest.mark.parametrize(
    "empty_feature",
    [
        "consumption_vs_usual",  # e.g. the home has no usual yet
        "voltage_gap_with_neighbours_vs_usual",  # e.g. none of its neighbours is usable
        "phase_unexplained_energy",  # e.g. a meter on its phase was caught, so it can't be computed
        "transformer_voltage_vs_usual",  # e.g. the transformer's reading on its phase is broken
    ],
)
def test_is_home_row_scorable_says_no_when_a_feature_is_empty(empty_feature):
    """A measured row the rules didn't untrust still can't be scored if any one of its features is empty."""
    # Prepare: H1, measured, with this one feature empty; the rules untrusted nobody.
    new_hour_home = pd.DataFrame([home_row("H1", "A")], columns=HOME_HOURS_COLUMNS)
    h1_forest_table_row = forest_table_row("H1")
    h1_forest_table_row[empty_feature] = np.nan  # this one feature couldn't be computed
    forest_table = pd.DataFrame([h1_forest_table_row], columns=FOREST_TABLE_COLUMNS)
    is_home_untrusted_by_rules = pd.Series([False])
    # Run.
    is_home_row_scorable = trust.is_home_row_scorable(new_hour_home, is_home_untrusted_by_rules, forest_table)
    # Check: no.
    assert list(is_home_row_scorable) == [False]


# =============================================================================
# TRUST STATES DURING LEARNING (home_trust_states_during_learning, phase_trust_states_during_learning)
# =============================================================================
def test_home_trust_states_during_learning_follow_the_four_cases():
    """Each home row gets the trust state of the first case that holds: 1. the rules untrusted it: untrusted;
    2. its reading is estimated: trusted; 3. the forest can score it: trusted (the learning rule);
    otherwise (a feature is empty): unchecked."""
    # Prepare: five homes at the new hour.
    #   H1: the rules untrusted it, and its reading is estimated too (case 1 comes first)
    #   H2: estimated, with an empty feature (an estimated reading is trusted all the same)
    #   H3: measured, every feature filled in
    #   H4: measured, but its phase_unexplained_energy is empty (e.g. a meter on its phase was caught)
    #   H5: measured, but it has no usual yet, so consumption_vs_usual is empty
    new_hour_home = pd.DataFrame(
        [
            home_row("H1", "A", estimated=1),
            home_row("H2", "A", estimated=1),
            home_row("H3", "B"),
            home_row("H4", "C"),
            home_row("H5", "C"),
        ],
        columns=HOME_HOURS_COLUMNS,
    )
    forest_table = pd.DataFrame(
        [
            forest_table_row("H1"),
            forest_table_row("H2", consumption_vs_usual=np.nan),
            forest_table_row("H3"),
            forest_table_row("H4", phase_unexplained_energy=np.nan),
            forest_table_row("H5", consumption_vs_usual=np.nan),
        ],
        columns=FOREST_TABLE_COLUMNS,
    )
    is_home_untrusted_by_rules = pd.Series([True, False, False, False, False])  # the rules untrusted H1 only
    # Run.
    home_trust_states_during_learning = trust.home_trust_states_during_learning(
        new_hour_home, is_home_untrusted_by_rules, forest_table
    )
    # Check, home by home: H1 untrusted (case 1), H2 trusted (case 2), H3 trusted (case 3), H4 and H5 unchecked.
    assert list(home_trust_states_during_learning) == ["untrusted", "trusted", "trusted", "unchecked", "unchecked"]


def test_phase_trust_states_during_learning_follow_the_three_cases():
    """Each phase row gets the trust state of the first case that holds: 1. the rules untrusted it: untrusted;
    2. its transformer voltage feature is empty (no usual voltage yet): unchecked; otherwise: trusted. Its
    unexplained energy plays no part: a phase row only feeds the transformer voltage usual."""
    # Prepare: three phases with one home each (each home row carries its phase's two phase features).
    #   A: the rules untrusted it (e.g. its transformer meter is broken), so both its phase features are empty
    #   B: it has no usual voltage yet (the first days), so its transformer voltage feature is empty
    #   C: its unexplained energy is empty (a meter on C was caught), but its transformer voltage feature is filled in
    new_hour_phase = pd.DataFrame([phase_row("A"), phase_row("B"), phase_row("C")], columns=PHASE_HOURS_COLUMNS)
    new_hour_home = pd.DataFrame(
        [home_row("H1", "A"), home_row("H2", "B"), home_row("H3", "C")], columns=HOME_HOURS_COLUMNS
    )
    forest_table = pd.DataFrame(
        [
            forest_table_row("H1", phase_unexplained_energy=np.nan, transformer_voltage_vs_usual=np.nan),
            forest_table_row("H2", transformer_voltage_vs_usual=np.nan),
            forest_table_row("H3", phase_unexplained_energy=np.nan),
        ],
        columns=FOREST_TABLE_COLUMNS,
    )
    is_phase_untrusted_by_rules = pd.Series([True, False, False])  # the rules untrusted phase A only
    # Run.
    phase_trust_states_during_learning = trust.phase_trust_states_during_learning(
        new_hour_phase, new_hour_home, is_phase_untrusted_by_rules, forest_table
    )
    # Check: A untrusted (case 1), B unchecked (case 2), C trusted (its empty unexplained energy plays no part).
    assert list(phase_trust_states_during_learning) == ["untrusted", "unchecked", "trusted"]
