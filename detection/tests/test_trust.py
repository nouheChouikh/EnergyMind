"""Tests for detector/trust.py: which home rows and phase rows may feed "usual" (plan step 4, the trust rule).

What it does: checks, on the made-up street of conftest.py, which homes and which phases the rules make untrusted
at the new hour, for each kind of finding and for several findings at once.

Why it exists: trust decides what the usuals learn from. If a caught reading slipped through as trusted, it would
feed a usual, and every later hour would be compared with a polluted "normal".

Choices:
  - Findings are planted by hand (with finding() below), so each test holds exactly the finding it needs. Trust reads
    only the findings, so the street's readings stay honest.
  - finding() and findings_table() are copied from test_features.py (the losses tests use them too): about ten lines,
    simpler than a shared helper file.
  - Test names start with what they test (test_untrusted_by_rules_...), so the test report reads in order.
  - Names say what a value holds: a yes/no for each row starts with is_; a set of names is named after what's in it
    (untrusted_homes). An answer keeps the full name of the function that gives it (is_home_untrusted_by_rules): the
    function is called through its file (trust.is_home_untrusted_by_rules), so the two names never clash.
"""

import pandas as pd  # tables

from detector import settings  # the scope and verdict names a finding is written with
from detector import trust  # the code being tested, called through its file (e.g. trust.is_home_untrusted_by_rules)
from detector.data_formats import RULE_FINDINGS_COLUMNS  # the findings table's columns
from detector.preparation.tables import build_phase_hours  # builds the phase table from the made-up street


# =============================================================================
# HELPERS
# =============================================================================
# The tests use the made-up street of conftest.py at Friday 12:00 (H1 and H2 on phase A, H3 and H4 on B, H5 and H6
# on C), with findings planted by hand. Trust reads only the findings, so the readings themselves stay honest.
TRUST_HOUR = pd.Timestamp(2019, 1, 4, 12)  # Friday 12:00, an hour of the made-up street
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
