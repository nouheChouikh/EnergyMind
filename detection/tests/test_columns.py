"""Tests for detector/columns/: the pieces the forest's columns are built from (plan steps 3 and 4).

What it does: checks each module of detector/columns/ on small made-up tables where the right answer is known in
advance. One section per module, in the order they're built: the losses regression, trust and the usual now; the
five forest columns join later.

Why it exists: the columns feed the forest, and the forest never complains about wrong inputs: it would quietly
learn wrong values as normal. So each piece is proven right on its own before the forest ever learns from it.

Choices:
  - The helpers at the top are shared by every section. Each section's own helpers sit under its own banner and
    are named after what they build, so sections never get mixed up.
  - Test names start with the function they test (test_losses_fit_..., test_usable_hours_...,
    test_expected_losses_..., test_untrusted_by_rules_..., test_consumption_usuals_...), so the test report reads
    function by function.
  - Names say what a value holds: a yes/no for each row starts with is_ or has_ (is_touched, is_home_untrusted);
    a set of names or a table of rows is named after what's in it (untrusted_homes, usable).
  - The losses tests build their tables from a known a and b per phase: the losses then follow a + b x E² exactly,
    so the fit must find exactly those numbers again.
  - To prove an hour is ignored, a test fills it with wild numbers: if the code used that hour, the fit would come
    out wrong.
"""

import numpy as np  # np.nan: an empty value
import pandas as pd  # tables
import pytest  # the test tool; @pytest.mark.parametrize runs one test once per case in a list

from detector import settings  # the scope and verdict names a finding is written with
from detector.columns.losses_regression import (  # the code being tested
    expected_losses,
    fit_losses_regression,
    usable_hours,
)
from detector.columns.trust import is_home_untrusted_by_rules, is_phase_untrusted_by_rules  # the code being tested
from detector.columns.usual import consumption_usuals, transformer_voltage_usuals  # the code being tested
from detector.data_formats import (  # the tables' columns
    HOME_HOURS_COLUMNS,
    HOME_MEMORY_COLUMNS,
    LAST_USUALS_COLUMNS,
    LOSSES_FIT_COLUMNS,
    PHASE_HOURS_COLUMNS,
    PHASE_MEMORY_COLUMNS,
    RULE_FINDINGS_COLUMNS,
)
from detector.preparation.tables import build_phase_hours  # builds the phase table from the made-up street


# =============================================================================
# SHARED HELPERS: used by every section
# =============================================================================
def phase_row(hour, phase, energy, homes_kwh, nb_homes_reporting=25, voltage=231.0, day_type="weekday"):
    """One row of the phase table: one phase at one hour, on a phase of 25 homes.

    energy: what the transformer sent on the phase (kWh). homes_kwh: what the phase's meters reported in total.
    nb_homes_reporting: how many of the 25 homes sent their kWh (25 means all of them: the hour is complete).
    voltage: the transformer's voltage on the phase. day_type: the hour's type of day.
    """
    return {
        "timestamp": hour,
        "phase": phase,
        "day_type": day_type,
        "transformer_kwh": energy,
        "transformer_voltage_v": voltage,
        "homes_kwh": homes_kwh,
        "nb_homes_reporting": nb_homes_reporting,
        "nb_homes_on_phase": 25,
    }


def with_extra_row(phase_hours, row):
    """A copy of the phase table with one extra row (made by phase_row) added at the end."""
    extra = pd.DataFrame([row], columns=PHASE_HOURS_COLUMNS)  # the extra row, as a one-row table
    return pd.concat([phase_hours, extra], ignore_index=True)  # stacked under the original rows


def finding(hour, scope, verdict, phase=None, home_id=None):
    """One rule finding, as a row of the findings table.

    Written out here rather than produced by the rules, so a test can plant exactly the finding it needs.
    phase: for a home, the home's phase; for the transformer, the broken phase; empty (None) for the whole street.
    """
    return {"timestamp": hour, "scope": scope, "phase": phase, "home_id": home_id, "verdict": verdict}


def findings_table(findings):
    """A findings table holding the findings in this list; findings_table([]) means no rule fired."""
    return pd.DataFrame(findings, columns=RULE_FINDINGS_COLUMNS)


# =============================================================================
# LOSSES REGRESSION (losses_regression.py): helpers
# =============================================================================
# The a and b used to build the made-up learning hours: the fit must find them again.
# Each phase gets different values, as on a real street where each phase carries different loads.
TRUE_LOSSES = {"A": (0.02, 0.001), "B": (0.01, 0.002), "C": (0.03, 0.0005)}

# The made-up learning period: 24 hours of one Monday (00:00 to 23:00).
LOSSES_LEARNING_HOURS = pd.date_range("2018-03-05 00:00", periods=24, freq="h")
# 10:00: the hour where the tests below plant a rule finding.
LOSSES_CAUGHT_HOUR = LOSSES_LEARNING_HOURS[10]
# Tuesday 00:00, just after the learning period: where the tests add an extra (incomplete) hour.
LOSSES_EXTRA_HOUR = LOSSES_LEARNING_HOURS[-1] + pd.Timedelta(hours=1)


def honest_learning_hours():
    """The made-up learning period: 24 complete, honest hours of all three phases.

    Each phase's losses follow its TRUE_LOSSES exactly (losses = a + b x E²), so the fit's answer is known.
    """
    rows = []
    for phase, (a, b) in TRUE_LOSSES.items():
        for step, hour in enumerate(LOSSES_LEARNING_HOURS):  # step: 0 at 00:00, 1 at 01:00 ... 23 at 23:00
            # What the transformer sent: 2.0 kWh at 00:00, rising by 0.5 kWh each hour to 13.5 kWh at 23:00.
            # Quiet and busy hours are both needed: they're what lets the fit tell a (fixed part) from b (growth).
            energy = 2.0 + 0.5 * step
            losses = a + b * energy**2  # what the cables lose at that energy
            # The homes' meters reported everything the cables didn't lose.
            rows.append(phase_row(hour, phase, energy, homes_kwh=energy - losses))
    return pd.DataFrame(rows, columns=PHASE_HOURS_COLUMNS)


def assert_fit_found_true_losses(losses_fit):
    """Checks the fit found TRUE_LOSSES again: one row per phase, A then B then C, each with its own a and b."""
    assert list(losses_fit.columns) == LOSSES_FIT_COLUMNS  # the agreed format: phase, a, b
    assert list(losses_fit["phase"]) == ["A", "B", "C"]
    for row in losses_fit.itertuples():  # one row at a time: row.phase, row.a, row.b
        true_a, true_b = TRUE_LOSSES[row.phase]
        # approx: equal up to tiny rounding errors, since computers can't store most decimals exactly.
        assert row.a == pytest.approx(true_a)
        assert row.b == pytest.approx(true_b)


def phases_kept_at(usable, hour):
    """The phases usable_hours kept at this hour, as a set, e.g. {"B", "C"}."""
    return set(usable.loc[usable["timestamp"] == hour, "phase"])


def fit_table_from_true_losses():
    """The fit's table as learning would return it, written out by hand from TRUE_LOSSES (no fitting involved)."""
    rows = []
    for phase, (a, b) in TRUE_LOSSES.items():
        rows.append({"phase": phase, "a": a, "b": b})
    return pd.DataFrame(rows, columns=LOSSES_FIT_COLUMNS)


def new_hour_phase_rows(energies):
    """The new hour's phase rows, one per phase in the order given, with these transformer energies.

    energies: e.g. {"C": 8.0, "A": 10.0}. The homes' sum plays no part in expected losses, so it's set equal to E.
    """
    hour = pd.Timestamp(2019, 1, 7, 18)  # any hour: expected losses don't depend on the time
    rows = []
    for phase, energy in energies.items():
        rows.append(phase_row(hour, phase, energy, homes_kwh=energy))
    return pd.DataFrame(rows, columns=PHASE_HOURS_COLUMNS)


# =============================================================================
# LOSSES REGRESSION (losses_regression.py): the fit, once at learning
# =============================================================================
# ----------------------------------------------------------------------------- honest hours
def test_losses_fit_finds_each_phases_a_and_b():
    """The fit's basic job: on hours that follow a + b x E² exactly, it finds each phase's own a and b again."""
    # Prepare: the honest learning period, and an empty findings table (no rule fired).
    phase_hours = honest_learning_hours()
    # Run: the fit.
    losses_fit = fit_losses_regression(phase_hours, findings_table([]))
    # Check: one row per phase, with the a and b the hours were built from.
    assert_fit_found_true_losses(losses_fit)


# ----------------------------------------------------------------------------- incomplete hours
# Each case below adds one incomplete hour on phase A, filled with wild numbers. If the fit used that hour, A's a
# and b would come out wrong; so finding the true values again proves the hour was ignored.
@pytest.mark.parametrize(  # runs the test once per case; ids are the cases' names in the test report
    "incomplete_row",
    [
        # Only 24 of the 25 homes sent their kWh: the homes' sum misses a home, so the losses would look like
        # 9 kWh (10 sent, 1 reported) instead of about 0.1.
        phase_row(LOSSES_EXTRA_HOUR, "A", energy=10.0, homes_kwh=1.0, nb_homes_reporting=24),
        # The transformer measured no energy (e.g. an outage): there's nothing to compare the homes with.
        phase_row(LOSSES_EXTRA_HOUR, "A", energy=0.0, homes_kwh=5.0),
    ],
    ids=["a home's kWh is missing", "no energy measured"],
)
def test_losses_fit_ignores_incomplete_hours(incomplete_row):
    """The fit only learns from complete hours: an incomplete one, even holding wild numbers, changes nothing."""
    # Prepare: the honest learning period, plus the incomplete hour.
    phase_hours = with_extra_row(honest_learning_hours(), incomplete_row)
    # Run: the fit.
    losses_fit = fit_losses_regression(phase_hours, findings_table([]))
    # Check: still the true a and b, as if the incomplete hour wasn't there.
    assert_fit_found_true_losses(losses_fit)


# ----------------------------------------------------------------------------- hours a rule caught
# Each case plants one rule finding at 10:00 and fills the phases it touches with a wild homes' sum. The fit must
# ignore exactly those phase-hours: a home's or the transformer's finding touches its phase, a street finding (no
# phase) touches all three.
@pytest.mark.parametrize(
    "rule_finding, touched_phases",
    [
        # H07, a home on phase A, sent an impossible reading: phase A is touched.
        (
            finding(LOSSES_CAUGHT_HOUR, settings.SCOPE_HOME, settings.VERDICT_IMPOSSIBLE, phase="A", home_id="H07"),
            ["A"],
        ),
        # The transformer's own reading on phase A is broken: phase A is touched.
        (
            finding(LOSSES_CAUGHT_HOUR, settings.SCOPE_TRANSFORMER, settings.VERDICT_REFERENCE_METER, phase="A"),
            ["A"],
        ),
        # Every meter of the street went silent (no phase: it concerns the whole street): all three are touched.
        (
            finding(LOSSES_CAUGHT_HOUR, settings.SCOPE_STREET, settings.VERDICT_MISSING_ALL),
            ["A", "B", "C"],
        ),
    ],
    ids=["a home on A", "the transformer on A", "the whole street"],
)
def test_losses_fit_ignores_hours_a_rule_caught(rule_finding, touched_phases):
    """The fit never learns from a reading a rule caught: those phase-hours, even holding wild numbers, change nothing."""
    # Prepare: at 10:00, on the phases the finding touches, the homes report 1 kWh when the transformer sent 7.
    phase_hours = honest_learning_hours()
    is_touched = (phase_hours["timestamp"] == LOSSES_CAUGHT_HOUR) & phase_hours["phase"].isin(touched_phases)
    phase_hours.loc[is_touched, "homes_kwh"] = 1.0
    # Run: the fit, with the finding.
    losses_fit = fit_losses_regression(phase_hours, findings_table([rule_finding]))
    # Check: still the true a and b.
    assert_fit_found_true_losses(losses_fit)


# ----------------------------------------------------------------------------- refusals
def test_losses_fit_refuses_losses_that_shrink_as_load_grows():
    """Losses that shrink as the load grows (b below 0) aren't physical: the fit stops and names the phase."""
    # Prepare: on phase B, the homes' sum is set so the losses go down as the energy goes up:
    # losses = 0.5 - 0.001 x E² (so b = -0.001).
    phase_hours = honest_learning_hours()
    is_on_b = phase_hours["phase"] == "B"
    energy = phase_hours.loc[is_on_b, "transformer_kwh"]
    phase_hours.loc[is_on_b, "homes_kwh"] = energy - (0.5 - 0.001 * energy**2)
    # Run and check: the fit stops with an error whose message names phase B.
    with pytest.raises(ValueError, match="phase B"):  # the test fails if no such error is raised
        fit_losses_regression(phase_hours, findings_table([]))


def test_losses_fit_refuses_a_phase_with_no_usable_hour():
    """A phase with no usable hour can't be fitted: the fit stops and names it, instead of a confusing error."""
    # Prepare: on phase C, one home never sent its kWh, so not a single hour on C is complete
    # (on a real street: a meter dead all year).
    phase_hours = honest_learning_hours()
    phase_hours.loc[phase_hours["phase"] == "C", "nb_homes_reporting"] = 24
    # Run and check: the fit stops with an error whose message names phase C.
    with pytest.raises(ValueError, match="phase C"):
        fit_losses_regression(phase_hours, findings_table([]))


# =============================================================================
# LOSSES REGRESSION (losses_regression.py): which hours the fit learns from
# =============================================================================
# The fit tests above can't see a clean hour dropped by mistake (the fit would still be exact), so these tests look
# directly at which phase-hours usable_hours keeps.
def test_usable_hours_home_finding_drops_only_its_phase():
    """A home's finding carries the home's phase: only that phase is dropped at that hour, the other two stay."""
    # Prepare: H07, a home on phase A, was caught stuck at 10:00.
    stuck = finding(LOSSES_CAUGHT_HOUR, settings.SCOPE_HOME, settings.VERDICT_STUCK, phase="A", home_id="H07")
    # Run: which phase-hours the fit may learn from.
    usable = usable_hours(honest_learning_hours(), findings_table([stuck]))
    # Check: at 10:00, B and C are kept and A is dropped; at 11:00 nothing was caught, so all three are kept.
    assert phases_kept_at(usable, LOSSES_CAUGHT_HOUR) == {"B", "C"}
    assert phases_kept_at(usable, LOSSES_LEARNING_HOURS[11]) == {"A", "B", "C"}


def test_usable_hours_street_finding_drops_all_three_phases():
    """A street finding has no phase: it concerns the whole street, so all three phases are dropped at that hour."""
    # Prepare: all the street's meters went silent at 10:00.
    silent = finding(LOSSES_CAUGHT_HOUR, settings.SCOPE_STREET, settings.VERDICT_MISSING_ALL)
    # Run: which phase-hours the fit may learn from.
    usable = usable_hours(honest_learning_hours(), findings_table([silent]))
    # Check: nothing is kept at 10:00; at 11:00 all three phases are kept.
    assert phases_kept_at(usable, LOSSES_CAUGHT_HOUR) == set()
    assert phases_kept_at(usable, LOSSES_LEARNING_HOURS[11]) == {"A", "B", "C"}


# =============================================================================
# LOSSES REGRESSION (losses_regression.py): the expected losses, live every hour
# =============================================================================
def test_expected_losses_follow_the_formula():
    """When energy was sent, each phase's expected losses are a + b x E², with that phase's own a and b."""
    # Prepare: the new hour's three phases, listed C, A, B on purpose: each answer must follow its own row's phase,
    # not the alphabetical order.
    new_hour_phase = new_hour_phase_rows({"C": 8.0, "A": 10.0, "B": 5.0})
    # Run: the expected losses, using the fit written out from TRUE_LOSSES.
    losses = expected_losses(new_hour_phase, fit_table_from_true_losses())
    # Check, row by row: C: 0.03 + 0.0005 x 64 = 0.062; A: 0.02 + 0.001 x 100 = 0.12; B: 0.01 + 0.002 x 25 = 0.06.
    assert list(losses) == pytest.approx([0.062, 0.12, 0.06])


def test_expected_losses_are_zero_when_no_energy_was_sent():
    """No energy sent, nothing lost: the answer is 0, not the fit's fixed part a (0.02 on phase A)."""
    # Prepare: phase A's transformer sent nothing this hour.
    new_hour_phase = new_hour_phase_rows({"A": 0.0})
    # Run and check: exactly 0.
    assert list(expected_losses(new_hour_phase, fit_table_from_true_losses())) == [0.0]


@pytest.mark.parametrize("broken_energy", [np.nan, -3.0], ids=["empty", "negative"])
def test_expected_losses_are_empty_for_a_broken_reading(broken_energy):
    """An empty or negative transformer reading is broken: there are no expected losses for it (an empty value)."""
    # Prepare: phase A's transformer reading is broken (empty in one case, negative in the other).
    new_hour_phase = new_hour_phase_rows({"A": broken_energy})
    # Run: the expected losses.
    losses = expected_losses(new_hour_phase, fit_table_from_true_losses())
    # Check: the answer is empty, rather than a number the formula would invent.
    assert losses.isna().all()


# =============================================================================
# TRUST (trust.py): helpers
# =============================================================================
# The tests use the made-up street of conftest.py at Friday 12:00 (H1 and H2 on phase A, H3 and H4 on B, H5 and H6
# on C), with findings planted by hand. Trust reads only the findings, so the readings themselves stay honest.
TRUST_HOUR = pd.Timestamp(2019, 1, 4, 12)  # Friday 12:00, an hour of the made-up street
ALL_HOMES = {"H1", "H2", "H3", "H4", "H5", "H6"}  # every home of the made-up street
ALL_PHASES = {"A", "B", "C"}


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
    is_home_untrusted = is_home_untrusted_by_rules(new_hour_home, rule_findings)
    is_phase_untrusted = is_phase_untrusted_by_rules(new_hour_phase, rule_findings)
    # Keep the names of the rows answered True.
    untrusted_homes = set(new_hour_home.loc[is_home_untrusted, "home_id"])
    untrusted_phases = set(new_hour_phase.loc[is_phase_untrusted, "phase"])
    return untrusted_homes, untrusted_phases


# =============================================================================
# TRUST (trust.py): what the rules make untrusted at the new hour
# =============================================================================
# ----------------------------------------------------------------------------- no finding
def test_untrusted_by_rules_nothing_without_findings(home_hours, transformer_readings):
    """When no rule fired, the rules untrust nothing: one answer per row, and every answer is False."""
    # Prepare: the street at Friday 12:00, and an empty findings table.
    new_hour_home, new_hour_phase = street_at_trust_hour(home_hours, transformer_readings)
    no_findings = findings_table([])
    # Run: both functions.
    is_home_untrusted = is_home_untrusted_by_rules(new_hour_home, no_findings)
    is_phase_untrusted = is_phase_untrusted_by_rules(new_hour_phase, no_findings)
    # Check: 6 answers for the 6 homes and 3 for the 3 phases, every one False.
    assert list(is_home_untrusted) == [False] * 6
    assert list(is_phase_untrusted) == [False] * 3


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
# USUAL (usual.py): helpers
# =============================================================================
# The tests use small hand-written memories: a few past rows, each with a value and a trust state, so every median
# can be checked by hand. The new hour is Tuesday 15 January 2019 at 18:00, a weekday.
USUAL_NEW_HOUR = pd.Timestamp(2019, 1, 15, 18)

# H1's five trusted weekday evenings at 18:00 before the new hour, and the kWh it used on each: their median is 0.7.
H1_EVENINGS = [
    (pd.Timestamp(2019, 1, 8, 18), 0.5),  # Tuesday
    (pd.Timestamp(2019, 1, 9, 18), 0.6),  # Wednesday
    (pd.Timestamp(2019, 1, 10, 18), 0.7),  # Thursday
    (pd.Timestamp(2019, 1, 11, 18), 0.8),  # Friday
    (pd.Timestamp(2019, 1, 14, 18), 0.9),  # Monday
]


def home_memory_row(hour, home_id, kwh, trust=settings.TRUST_TRUSTED, day_type="weekday"):
    """One past row of the home memory. Only the hour, the home, the type of day, the kWh and the trust matter
    for the usual; the other columns hold ordinary values."""
    return {
        "timestamp": hour,
        "home_id": home_id,
        "phase": "A",
        "cable": 1,
        "contracted_kw": 3.3,
        "day_type": day_type,
        "kwh": kwh,
        "voltage_v": 230.0,
        "current_a": 1.0,
        "estimated": 0,
        "trust": trust,
    }


def h1_evening_rows():
    """H1_EVENINGS as past rows of the home memory, all trusted."""
    rows = []
    for hour, kwh in H1_EVENINGS:
        rows.append(home_memory_row(hour, "H1", kwh))
    return rows


def home_memory(rows):
    """The home memory holding these past rows (made by home_memory_row)."""
    return pd.DataFrame(rows, columns=HOME_MEMORY_COLUMNS)


def new_hour_home_rows(home_ids):
    """The new hour's home rows (Tuesday 18:00, a weekday), one per home, in the order given."""
    rows = []
    for home_id in home_ids:
        rows.append(home_memory_row(USUAL_NEW_HOUR, home_id, kwh=0.65))
    # The home table's columns have no trust (the new hour is what's being checked), so it's left out here.
    return pd.DataFrame(rows, columns=HOME_HOURS_COLUMNS)


def phase_memory_row(hour, phase, voltage, trust=settings.TRUST_TRUSTED):
    """One past row of the phase memory: a weekday phase row (made by phase_row) plus its trust state."""
    row = phase_row(hour, phase, energy=10.0, homes_kwh=9.9, voltage=voltage)
    row["trust"] = trust
    return row


def last_usual_row(home_or_phase, usual, hour_of_day=18, usual_of=settings.USUAL_OF_CONSUMPTION):
    """One remembered usual, for weekdays at hour_of_day (18:00 unless told otherwise)."""
    return {
        "usual_of": usual_of,
        "home_or_phase": home_or_phase,
        "hour_of_day": hour_of_day,
        "day_type": "weekday",
        "usual": usual,
    }


def last_usuals_table(rows):
    """The last usuals table holding these remembered usuals; last_usuals_table([]) means none yet (the start)."""
    return pd.DataFrame(rows, columns=LAST_USUALS_COLUMNS)


def remembered_usuals(last_usuals, home_or_phase, hour_of_day=18, usual_of=settings.USUAL_OF_CONSUMPTION):
    """The values the last usuals table remembers for this home (or phase), for weekdays at hour_of_day, as a list.

    A list rather than one value, so a test can check there's exactly one (a new value replaces, never adds).
    """
    is_this_slot = (
        (last_usuals["usual_of"] == usual_of)
        & (last_usuals["home_or_phase"] == home_or_phase)
        & (last_usuals["hour_of_day"] == hour_of_day)
        & (last_usuals["day_type"] == "weekday")
    )
    return list(last_usuals.loc[is_this_slot, "usual"])


# =============================================================================
# USUAL (usual.py): each home's usual consumption
# =============================================================================
# ----------------------------------------------------------------------------- the median
def test_consumption_usuals_are_the_median_of_trusted_values():
    """With at least 3 trusted values at the same hour, on the same type of day, the usual is their median."""
    # Prepare: H1's five trusted weekday evenings (0.5 to 0.9 kWh), and nothing remembered yet.
    memory = home_memory(h1_evening_rows())
    # Run: H1's usual at Tuesday 18:00.
    usuals, _updated_last_usuals = consumption_usuals(new_hour_home_rows(["H1"]), memory, last_usuals_table([]))
    # Check: the median of 0.5, 0.6, 0.7, 0.8 and 0.9 is 0.7.
    assert list(usuals) == pytest.approx([0.7])


# ----------------------------------------------------------------------------- rows that don't count
# Each case adds one past row that must be left out, holding a wild 50 kWh. If the code counted it, the median of
# the six values would be 0.75 instead of 0.7; so finding 0.7 again proves the row was left out.
@pytest.mark.parametrize(
    "left_out_row",
    [
        home_memory_row(pd.Timestamp(2019, 1, 7, 18), "H1", 50.0, trust=settings.TRUST_UNTRUSTED),  # a Monday
        home_memory_row(pd.Timestamp(2019, 1, 7, 18), "H1", 50.0, trust=settings.TRUST_UNCHECKED),  # a Monday
        home_memory_row(pd.Timestamp(2019, 1, 14, 17), "H1", 50.0),  # Monday at 17:00, not 18:00
        home_memory_row(pd.Timestamp(2019, 1, 12, 18), "H1", 50.0, day_type="weekend_or_holiday"),  # a Saturday
        home_memory_row(pd.Timestamp(2018, 12, 17, 18), "H1", 50.0),  # 29 days before the new hour
    ],
    ids=["untrusted", "unchecked", "another hour", "the other type of day", "older than 28 days"],
)
def test_consumption_usuals_leave_out_rows_that_dont_count(left_out_row):
    """Only trusted rows, at the same hour, on the same type of day, within the last 28 days feed the usual."""
    # Prepare: H1's five trusted evenings, plus the row that must be left out.
    memory = home_memory(h1_evening_rows() + [left_out_row])
    # Run.
    usuals, _updated_last_usuals = consumption_usuals(new_hour_home_rows(["H1"]), memory, last_usuals_table([]))
    # Check: still 0.7.
    assert list(usuals) == pytest.approx([0.7])


# ----------------------------------------------------------------------------- too few trusted values
def test_consumption_usuals_fall_back_on_the_remembered_usual():
    """With fewer than 3 trusted values (e.g. during a long theft), the usual is the one remembered for that hour."""
    # Prepare: only the first two of H1's evenings, and 0.55 remembered for weekdays at 18:00.
    memory = home_memory(h1_evening_rows()[:2])
    last_usuals = last_usuals_table([last_usual_row("H1", 0.55)])
    # Run.
    usuals, _updated_last_usuals = consumption_usuals(new_hour_home_rows(["H1"]), memory, last_usuals)
    # Check: 0.55, the remembered usual.
    assert list(usuals) == pytest.approx([0.55])


def test_consumption_usuals_are_empty_when_none_was_ever_computed():
    """With fewer than 3 trusted values and nothing remembered (the first weeks), there's no usual yet: empty."""
    # Prepare: only the first two of H1's evenings, and nothing remembered.
    memory = home_memory(h1_evening_rows()[:2])
    # Run.
    usuals, _updated_last_usuals = consumption_usuals(new_hour_home_rows(["H1"]), memory, last_usuals_table([]))
    # Check: empty.
    assert usuals.isna().all()


# ----------------------------------------------------------------------------- remembering
def test_consumption_usuals_remember_each_new_median():
    """A new median overwrites the remembered value for the same home, hour and type of day; other values stay."""
    # Prepare: H1's five evenings; H1 had 0.55 remembered for weekdays at 18:00, and 0.40 for weekdays at 17:00.
    memory = home_memory(h1_evening_rows())
    last_usuals = last_usuals_table([last_usual_row("H1", 0.55), last_usual_row("H1", 0.40, hour_of_day=17)])
    # Run.
    _usuals, updated_last_usuals = consumption_usuals(new_hour_home_rows(["H1"]), memory, last_usuals)
    # Check: 18:00 now remembers the new median 0.7 (one value, not two); 17:00 still remembers 0.40.
    assert remembered_usuals(updated_last_usuals, "H1", hour_of_day=18) == pytest.approx([0.7])
    assert remembered_usuals(updated_last_usuals, "H1", hour_of_day=17) == pytest.approx([0.40])


# ----------------------------------------------------------------------------- several homes
def test_consumption_usuals_follow_the_new_hours_rows():
    """Each home gets its own usual, in the order of the new hour's rows."""
    # Prepare: H1's five evenings (median 0.7), and H2's three evenings 0.2, 0.3 and 0.4 (median 0.3).
    rows = h1_evening_rows()
    rows.append(home_memory_row(pd.Timestamp(2019, 1, 9, 18), "H2", 0.2))
    rows.append(home_memory_row(pd.Timestamp(2019, 1, 10, 18), "H2", 0.3))
    rows.append(home_memory_row(pd.Timestamp(2019, 1, 11, 18), "H2", 0.4))
    # Run: the new hour lists H2 first, then H1.
    new_hour_home = new_hour_home_rows(["H2", "H1"])
    usuals, _updated_last_usuals = consumption_usuals(new_hour_home, home_memory(rows), last_usuals_table([]))
    # Check: H2's usual first, then H1's.
    assert list(usuals) == pytest.approx([0.3, 0.7])


# =============================================================================
# USUAL (usual.py): each phase's usual transformer voltage
# =============================================================================
def test_transformer_voltage_usuals_are_each_phases_median():
    """The transformer voltage's usual works the same way, with one usual per phase."""
    # Prepare: three trusted weekday evenings for phases A and B, each phase with its own voltages.
    rows = []
    for day, voltage_on_a, voltage_on_b in [(9, 231.0, 229.0), (10, 232.0, 230.0), (11, 233.0, 231.0)]:
        evening = pd.Timestamp(2019, 1, day, 18)
        rows.append(phase_memory_row(evening, "A", voltage_on_a))
        rows.append(phase_memory_row(evening, "B", voltage_on_b))
    phase_memory = pd.DataFrame(rows, columns=PHASE_MEMORY_COLUMNS)
    new_hour_phase = pd.DataFrame(
        [phase_row(USUAL_NEW_HOUR, "A", 10.0, 9.9), phase_row(USUAL_NEW_HOUR, "B", 10.0, 9.9)],
        columns=PHASE_HOURS_COLUMNS,
    )
    # Run.
    usuals, updated_last_usuals = transformer_voltage_usuals(new_hour_phase, phase_memory, last_usuals_table([]))
    # Check: A's median is 232.0 and B's is 230.0; A's is remembered under the transformer voltage's name.
    assert list(usuals) == pytest.approx([232.0, 230.0])
    remembered_on_a = remembered_usuals(updated_last_usuals, "A", usual_of=settings.USUAL_OF_TRANSFORMER_VOLTAGE)
    assert remembered_on_a == pytest.approx([232.0])
