"""Tests for detector/features/: what turns the home and phase tables into the forest's features (plan steps 3, 4).

What it does: checks each module of detector/features/ on small made-up tables where the right answer is known in
advance. One section per module, in the order they're built: the losses regression, the usual, the forest's
features (forest_table.py), the gap stage they read (voltage_gap_with_neighbours.py), then the forest table.

Why it exists: the features feed the forest, and the forest never complains about wrong inputs: it would quietly
learn wrong values as normal. So each piece is proven right on its own before the forest ever learns from it.

Choices:
  - The helpers at the top are shared by every section. Each section's own helpers sit under its own banner and
    are named after what they build, so sections never get mixed up.
  - Trust's tests live in test_trust.py: trust.py sits at the top of detector/, since it isn't something a feature
    is built from.
  - Test names start with the function they test (test_losses_fit_..., test_usable_hours_...,
    test_expected_losses_..., test_consumption_usuals_..., test_consumption_vs_usual_...,
    test_is_unexplained_energy_computable_..., test_voltage_gap_with_neighbours_..., test_build_forest_table_...),
    so the test report reads function by function.
  - Names say what a value holds: a yes/no for each row starts with is_ or has_ (is_touched,
    is_home_untrusted_by_rules); a table of rows is named after what's in it (usable: the usable rows). An answer
    keeps the full name of the function that gives it: where a test calls that function, it checks the answer
    directly instead of storing it under a shorter name.
  - The losses tests build their tables from a known a and b per phase: the losses then follow a + b x E² exactly,
    so the fit must find exactly those numbers again.
  - To prove an hour is ignored, a test fills it with wild numbers: if the code used that hour, the fit would come
    out wrong.
"""

import numpy as np  # np.nan: an empty value
import pandas as pd  # tables
import pytest  # the test tool; @pytest.mark.parametrize runs one test once per case in a list

from detector import settings  # the names findings and trust states are written with, and the usual's minimum
from detector.features.forest_table import (  # the code being tested
    build_forest_table,
    consumption_vs_usual,
    is_unexplained_energy_computable,
    phase_unexplained_energy,
    transformer_voltage_vs_usual,
    voltage_gap_with_neighbours_vs_usual,
)
from detector.features.losses_regression import (  # the code being tested
    expected_losses,
    fit_losses_regression,
    usable_hours,
)
from detector.features.usual import (  # the code being tested
    consumption_usuals,
    transformer_voltage_usuals,
    voltage_gap_with_neighbours_usuals,
)
from detector.features.voltage_gap_with_neighbours import voltage_gap_with_neighbours  # the code being tested
from detector.data_formats import (  # the tables' columns
    HOME_HOURS_COLUMNS,
    HOME_MEMORY_COLUMNS,
    LAST_USUALS_COLUMNS,
    LOSSES_FIT_COLUMNS,
    PHASE_HOURS_COLUMNS,
    PHASE_MEMORY_COLUMNS,
    RULE_FINDINGS_COLUMNS,
)


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


def home_memory_row(hour, home_id, kwh, trust=settings.TRUST_TRUSTED, day_type="weekday",
                    voltage_gap_with_neighbours=0.0):
    """One past row of the home memory. Only the hour, the home, the type of day, the kWh (or, for the gap's usual,
    the voltage gap with the neighbours) and the trust matter for the usual; the other columns hold ordinary values."""
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
        "voltage_gap_with_neighbours": voltage_gap_with_neighbours,
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
    is_same_usual_home_hour_and_day_type = (
        (last_usuals["usual_of"] == usual_of)
        & (last_usuals["home_or_phase"] == home_or_phase)
        & (last_usuals["hour_of_day"] == hour_of_day)
        & (last_usuals["day_type"] == "weekday")
    )
    return list(last_usuals.loc[is_same_usual_home_hour_and_day_type, "usual"])


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
        home_memory_row(pd.Timestamp(2019, 1, 14, 17), "H1", 50.0),  # Monday at 17:00, not 18:00
        home_memory_row(pd.Timestamp(2019, 1, 12, 18), "H1", 50.0, day_type="weekend_or_holiday"),  # a Saturday
        home_memory_row(pd.Timestamp(2018, 12, 17, 18), "H1", 50.0),  # 29 days before the new hour
    ],
    ids=["untrusted", "another hour", "the other type of day", "older than 28 days"],
)
def test_consumption_usuals_leave_out_rows_that_dont_count(left_out_row):
    """An untrusted row, another hour, the other type of day, or a row older than 28 days never feeds the usual."""
    # Prepare: H1's five trusted evenings, plus the row that must be left out.
    memory = home_memory(h1_evening_rows() + [left_out_row])
    # Run.
    usuals, _updated_last_usuals = consumption_usuals(new_hour_home_rows(["H1"]), memory, last_usuals_table([]))
    # Check: still 0.7.
    assert list(usuals) == pytest.approx([0.7])


# ----------------------------------------------------------------------------- unchecked rows
# An unchecked row: no rule caught it, but nobody could check it (one of its features was empty). It counts only
# toward a home's very first usual: the first-usual exception.
def test_consumption_usuals_leave_out_unchecked_rows_once_a_usual_is_remembered():
    """Once a home has a remembered usual, only trusted rows count: an unchecked row is left out."""
    # Prepare: H1's five trusted evenings (median 0.7), an unchecked Monday holding a wild 50 kWh, and a usual
    # already remembered for H1 at weekday 18:00 (0.55).
    unchecked_monday = home_memory_row(pd.Timestamp(2019, 1, 7, 18), "H1", 50.0, trust=settings.TRUST_UNCHECKED)
    memory = home_memory(h1_evening_rows() + [unchecked_monday])
    last_usuals = last_usuals_table([last_usual_row("H1", 0.55)])
    # Run.
    usuals, _updated_last_usuals = consumption_usuals(new_hour_home_rows(["H1"]), memory, last_usuals)
    # Check: still 0.7. If the unchecked Monday counted, the median of the six values would be 0.75.
    assert list(usuals) == pytest.approx([0.7])


def test_consumption_usuals_count_unchecked_rows_for_the_very_first_usual():
    """While nothing is remembered for a home, its unchecked rows count too. Without this, its first usual could
    never form: with no usual, a row can't be scored, so it's unchecked, so it wouldn't count."""
    # Prepare: nothing remembered for H1 yet. Its first three weekday evenings were unchecked (0.5, 0.6 and 0.7 kWh);
    # the rules untrusted a fourth one (a wild 50 kWh): an untrusted row never counts, even for a first usual.
    memory = home_memory([
        home_memory_row(pd.Timestamp(2019, 1, 8, 18), "H1", 0.5, trust=settings.TRUST_UNCHECKED),  # Tuesday
        home_memory_row(pd.Timestamp(2019, 1, 9, 18), "H1", 0.6, trust=settings.TRUST_UNCHECKED),  # Wednesday
        home_memory_row(pd.Timestamp(2019, 1, 10, 18), "H1", 0.7, trust=settings.TRUST_UNCHECKED),  # Thursday
        home_memory_row(pd.Timestamp(2019, 1, 11, 18), "H1", 50.0, trust=settings.TRUST_UNTRUSTED),  # Friday
    ])
    # Run: H1's usual at Tuesday 15 January, 18:00.
    usuals, updated_last_usuals = consumption_usuals(new_hour_home_rows(["H1"]), memory, last_usuals_table([]))
    # Check: the median of 0.5, 0.6 and 0.7 is 0.6, and it's remembered (from now on, only trusted rows count).
    assert list(usuals) == pytest.approx([0.6])
    assert remembered_usuals(updated_last_usuals, "H1") == pytest.approx([0.6])


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


# =============================================================================
# USUAL (usual.py): each home's usual voltage gap with its neighbours
# =============================================================================
def test_voltage_gap_with_neighbours_usuals_are_the_median_of_past_gaps():
    """The gap's usual works like the other two: the median of the home's past trusted gaps, read from the memory."""
    # Prepare: H1's gap with its neighbours on three trusted weekday evenings: -2, -3 and -4 V (it sits far down its
    # cable). Its kWh is 0.6 every evening: if the code read the kWh by mistake, the usual would come out 0.6.
    rows = []
    for day, gap in [(9, -2.0), (10, -3.0), (11, -4.0)]:
        rows.append(home_memory_row(pd.Timestamp(2019, 1, day, 18), "H1", 0.6, voltage_gap_with_neighbours=gap))
    # Run: H1's usual gap at Tuesday 18:00.
    usuals, updated_last_usuals = voltage_gap_with_neighbours_usuals(
        new_hour_home_rows(["H1"]), home_memory(rows), last_usuals_table([])
    )
    # Check: the median of -2, -3 and -4 is -3, and it's remembered under the gap's own name.
    assert list(usuals) == pytest.approx([-3.0])
    remembered = remembered_usuals(updated_last_usuals, "H1", usual_of=settings.USUAL_OF_VOLTAGE_GAP_WITH_NEIGHBOURS)
    assert remembered == pytest.approx([-3.0])


# =============================================================================
# FOREST FEATURES (forest_table.py): helpers
# =============================================================================
# The new hour every forest feature test checks: Friday 18 January 2019 at 18:00, a weekday evening.
FEATURES_HOUR = pd.Timestamp(2019, 1, 18, 18)


def home_row_at_features_hour(home_id, phase, kwh, estimated=0, cable=1, voltage=230.0):
    """One home row of the new hour (FEATURES_HOUR).

    The home, its cable and phase, its kWh, its voltage and its estimated flag (1 = the utility filled the value in
    itself) are what the features read; the other columns hold ordinary values.
    """
    return {
        "timestamp": FEATURES_HOUR,
        "home_id": home_id,
        "phase": phase,
        "cable": cable,
        "contracted_kw": 3.3,
        "day_type": "weekday",
        "kwh": kwh,
        "voltage_v": voltage,
        "current_a": 1.0,
        "estimated": estimated,
    }


def home_table(rows):
    """The new hour's home rows: a table holding these rows (made by home_row_at_features_hour), in the order given."""
    return pd.DataFrame(rows, columns=HOME_HOURS_COLUMNS)


def phase_table(rows):
    """The new hour's phase rows: a table holding these rows (made by phase_row), in the order given."""
    return pd.DataFrame(rows, columns=PHASE_HOURS_COLUMNS)


def yes_no(answers):
    """One yes/no per row, in the rows' order: yes_no([False, True]) is two rows, and only the second says yes.

    Stands in for answers another function would give: the rules' (is_home_untrusted_by_rules,
    is_phase_untrusted_by_rules) or is_unexplained_energy_computable's.
    """
    return pd.Series(answers)


def unexplained_energy_inputs():
    """is_unexplained_energy_computable's four inputs for an honest hour, where it says yes on every phase.

    Phases A, B and C, each with two homes that sent measured readings (A: H1 and H2; B: H3 and H4; C: H5 and H6),
    and the transformer measured 40 kWh on each phase. The rules untrusted nobody.
    Only two homes per phase are listed: is_unexplained_energy_computable looks at each home's trust and estimated
    flag, not at the phase's sums, so the sums (39 kWh reported) don't need to add up from these two homes.
    Returns the phase rows, the home rows, and the rules' two answers (one per home row, one per phase row).
    """
    new_hour_phase = phase_table([
        phase_row(FEATURES_HOUR, "A", energy=40.0, homes_kwh=39.0),
        phase_row(FEATURES_HOUR, "B", energy=40.0, homes_kwh=39.0),
        phase_row(FEATURES_HOUR, "C", energy=40.0, homes_kwh=39.0),
    ])
    new_hour_home = home_table([
        home_row_at_features_hour("H1", "A", kwh=0.5),
        home_row_at_features_hour("H2", "A", kwh=0.5),
        home_row_at_features_hour("H3", "B", kwh=0.5),
        home_row_at_features_hour("H4", "B", kwh=0.5),
        home_row_at_features_hour("H5", "C", kwh=0.5),
        home_row_at_features_hour("H6", "C", kwh=0.5),
    ])
    is_home_untrusted_by_rules = yes_no([False, False, False, False, False, False])  # one per home: nobody untrusted
    is_phase_untrusted_by_rules = yes_no([False, False, False])  # one answer per phase: no phase untrusted
    return new_hour_phase, new_hour_home, is_home_untrusted_by_rules, is_phase_untrusted_by_rules


# =============================================================================
# FOREST FEATURES (forest_table.py): consumption_vs_usual
# =============================================================================
def test_consumption_vs_usual_divides_kwh_by_the_usual():
    """The feature's basic job: this hour's kWh divided by the home's usual kWh."""
    # Prepare: H1 used 0.9 kWh and H2 0.3 kWh; both usually use 0.6 kWh at this hour. The rules untrusted nobody.
    new_hour_home = home_table([home_row_at_features_hour("H1", "A", 0.9), home_row_at_features_hour("H2", "B", 0.3)])
    usual_kwh = pd.Series([0.6, 0.6])  # one usual per home row, as consumption_usuals would give them
    # Run.
    ratios = consumption_vs_usual(new_hour_home, usual_kwh, yes_no([False, False]))
    # Check: 0.9 ÷ 0.6 = 1.5 (50% more than usual) and 0.3 ÷ 0.6 = 0.5 (half the usual).
    assert list(ratios) == pytest.approx([1.5, 0.5])


def test_consumption_vs_usual_raises_a_tiny_usual_to_the_minimum():
    """A usual below the minimum (in settings.py) counts as the minimum, so a tiny night usual can't make the ratio
    jump, and a usual of 0 can't make it divide by zero."""
    # Prepare: at night H1 usually uses 0.01 kWh and H2 0 kWh, both below the minimum; tonight they used 0.08 and 0.1.
    new_hour_home = home_table([home_row_at_features_hour("H1", "A", 0.08), home_row_at_features_hour("H2", "B", 0.1)])
    usual_kwh = pd.Series([0.01, 0.0])
    # Run.
    ratios = consumption_vs_usual(new_hour_home, usual_kwh, yes_no([False, False]))
    # Check: both are divided by the minimum instead. With today's 0.05 kWh: 0.08 ÷ 0.05 = 1.6 (not 0.08 ÷ 0.01 = 8)
    # and 0.1 ÷ 0.05 = 2.0.
    minimum = settings.CONSUMPTION_USUAL_MINIMUM_KWH
    assert list(ratios) == pytest.approx([0.08 / minimum, 0.1 / minimum])


def test_consumption_vs_usual_is_empty_without_a_usual_or_a_kwh():
    """No usual yet (a home's first days) or no kWh (nothing arrived): nothing to compare, so the feature is empty."""
    # Prepare: H1 has no usual yet; H2 sent no kWh; H3 has both (0.9 kWh, usual 0.6).
    new_hour_home = home_table([
        home_row_at_features_hour("H1", "A", 0.9),
        home_row_at_features_hour("H2", "B", np.nan),  # np.nan: an empty value
        home_row_at_features_hour("H3", "C", 0.9),
    ])
    usual_kwh = pd.Series([np.nan, 0.6, 0.6])
    # Run.
    ratios = consumption_vs_usual(new_hour_home, usual_kwh, yes_no([False, False, False]))
    # Check: H1 and H2 are empty (isna: "is it empty?"); H3 still gets 0.9 ÷ 0.6 = 1.5.
    assert list(ratios.isna()) == [True, True, False]
    assert ratios.iloc[2] == pytest.approx(1.5)  # iloc[2]: the third row's value


def test_consumption_vs_usual_is_empty_for_a_home_the_rules_untrusted():
    """A reading the rules untrusted (e.g. caught as stuck, or covered by an outage) gets no value: it's broken, and a
    broken value must never look real to the forest."""
    # Prepare: H1 is frozen on 0.9 kWh and the stuck rule caught it; H2 used 0.9 kWh too, honestly. Both usually use
    # 0.6 kWh. The rules' answers: H1 untrusted, H2 not.
    new_hour_home = home_table([home_row_at_features_hour("H1", "A", 0.9), home_row_at_features_hour("H2", "B", 0.9)])
    usual_kwh = pd.Series([0.6, 0.6])
    # Run.
    ratios = consumption_vs_usual(new_hour_home, usual_kwh, yes_no([True, False]))
    # Check: H1 is empty; H2 still gets 1.5.
    assert list(ratios.isna()) == [True, False]
    assert ratios.iloc[1] == pytest.approx(1.5)


# =============================================================================
# FOREST FEATURES (forest_table.py): is_unexplained_energy_computable
# =============================================================================
def test_is_unexplained_energy_computable_when_every_number_is_measured_and_trusted():
    """On an honest hour every phase's unexplained energy can be computed: one yes per phase."""
    # Prepare: the honest hour.
    new_hour_phase, new_hour_home, is_home_untrusted_by_rules, is_phase_untrusted_by_rules = unexplained_energy_inputs()
    # Run and check: is_unexplained_energy_computable answers yes for A, B and C.
    assert list(is_unexplained_energy_computable(
        new_hour_phase, new_hour_home, is_home_untrusted_by_rules, is_phase_untrusted_by_rules
    )) == [True, True, True]


# Each case breaks one thing on phase B only. B's unexplained energy must become uncomputable, and A's and C's must
# stay computable: a problem on one phase never blinds the other two.
@pytest.mark.parametrize(
    "what_breaks",
    [
        "a home the rules untrusted",
        "an estimated reading",
        "the phase row untrusted",
        "no energy measured",
        "an empty energy reading",
    ],
)
def test_is_unexplained_energy_computable_says_no_on_that_phase_only(what_breaks):
    """Any number in B's unexplained energy that isn't a real, trusted measurement makes it uncomputable (on B only)."""
    new_hour_phase, new_hour_home, is_home_untrusted_by_rules, is_phase_untrusted_by_rules = unexplained_energy_inputs()
    is_h3 = new_hour_home["home_id"] == "H3"  # H3's row (H3 is on phase B)
    is_phase_b = new_hour_phase["phase"] == "B"  # phase B's row
    # Prepare: break one thing on phase B.
    if what_breaks == "a home the rules untrusted":
        is_home_untrusted_by_rules.loc[is_h3] = True  # e.g. the stuck rule caught H3
    elif what_breaks == "an estimated reading":
        new_hour_home.loc[is_h3, "estimated"] = 1  # the utility filled H3's value in itself
    elif what_breaks == "the phase row untrusted":
        is_phase_untrusted_by_rules.loc[is_phase_b] = True  # e.g. the transformer's meter on B is broken
    elif what_breaks == "no energy measured":
        new_hour_phase.loc[is_phase_b, "transformer_kwh"] = 0.0  # the transformer sent nothing on B
    elif what_breaks == "an empty energy reading":
        new_hour_phase.loc[is_phase_b, "transformer_kwh"] = np.nan  # the transformer's value for B never arrived
    # Run and check: is_unexplained_energy_computable answers no for B only.
    assert list(is_unexplained_energy_computable(
        new_hour_phase, new_hour_home, is_home_untrusted_by_rules, is_phase_untrusted_by_rules
    )) == [True, False, True]


# =============================================================================
# FOREST FEATURES (forest_table.py): phase_unexplained_energy
# =============================================================================
def test_phase_unexplained_energy_follows_the_formula():
    """(transformer energy − its homes' kWh − expected losses) ÷ transformer energy × 100, phase by phase."""
    # Prepare: three phases whose unexplained energy can all be computed.
    #   A: 40 kWh sent, 39 reported, 0.6 expected losses: 0.4 kWh unexplained, 1% of 40 (energy missing)
    #   B: 20 kWh sent, 20.1 reported, 0.2 expected losses: -0.3 kWh unexplained, -1.5% of 20 (energy appearing)
    #   C: 10 kWh sent, 9.95 reported, 0.05 expected losses: 0 kWh unexplained, 0% (everything adds up)
    new_hour_phase = phase_table([
        phase_row(FEATURES_HOUR, "A", energy=40.0, homes_kwh=39.0),
        phase_row(FEATURES_HOUR, "B", energy=20.0, homes_kwh=20.1),
        phase_row(FEATURES_HOUR, "C", energy=10.0, homes_kwh=9.95),
    ])
    expected_losses_kwh = pd.Series([0.6, 0.2, 0.05])  # one per phase row, as expected_losses would give them
    # Run.
    percentages = phase_unexplained_energy(new_hour_phase, expected_losses_kwh, yes_no([True, True, True]))
    # Check.
    assert list(percentages) == pytest.approx([1.0, -1.5, 0.0])


def test_phase_unexplained_energy_is_empty_where_it_cant_be_computed():
    """Where is_unexplained_energy_computable said no, the feature is empty: unexplained energy built on a wrong or
    missing number must never look real to the forest."""
    # Prepare: A is honest (1%, as above); the transformer measured nothing on B (0 kWh, so its expected losses are 0),
    # and is_unexplained_energy_computable said no for it. With 0 kWh, the % would divide by zero: it must stay
    # silent and empty.
    new_hour_phase = phase_table([
        phase_row(FEATURES_HOUR, "A", energy=40.0, homes_kwh=39.0),
        phase_row(FEATURES_HOUR, "B", energy=0.0, homes_kwh=0.0),
    ])
    expected_losses_kwh = pd.Series([0.6, 0.0])
    # Run.
    percentages = phase_unexplained_energy(new_hour_phase, expected_losses_kwh, yes_no([True, False]))
    # Check: A still 1%; B empty.
    assert percentages.iloc[0] == pytest.approx(1.0)
    assert list(percentages.isna()) == [False, True]


# =============================================================================
# FOREST FEATURES (forest_table.py): transformer_voltage_vs_usual
# =============================================================================
def test_transformer_voltage_vs_usual_subtracts_the_usual():
    """The feature's basic job: the transformer's voltage on each phase minus its usual, in volts."""
    # Prepare: the transformer reads 226 V on A, 238 V on B and 231 V on C; each phase usually reads 231 V.
    new_hour_phase = phase_table([
        phase_row(FEATURES_HOUR, "A", energy=40.0, homes_kwh=39.0, voltage=226.0),
        phase_row(FEATURES_HOUR, "B", energy=40.0, homes_kwh=39.0, voltage=238.0),
        phase_row(FEATURES_HOUR, "C", energy=40.0, homes_kwh=39.0, voltage=231.0),
    ])
    usual_voltage = pd.Series([231.0, 231.0, 231.0])  # one per phase row, as transformer_voltage_usuals would give
    # Run.
    differences = transformer_voltage_vs_usual(new_hour_phase, usual_voltage, yes_no([False, False, False]))
    # Check: A 5 V below usual (a small sag), B 7 V above (a small surge), C exactly as usual.
    assert list(differences) == pytest.approx([-5.0, 7.0, 0.0])


def test_transformer_voltage_vs_usual_is_empty_without_a_usual():
    """A phase with no usual yet (the first days of the data) has nothing to compare with: its feature is empty."""
    # Prepare: A has a usual of 231 V; B has none yet.
    new_hour_phase = phase_table([
        phase_row(FEATURES_HOUR, "A", energy=40.0, homes_kwh=39.0, voltage=226.0),
        phase_row(FEATURES_HOUR, "B", energy=40.0, homes_kwh=39.0, voltage=226.0),
    ])
    usual_voltage = pd.Series([231.0, np.nan])
    # Run.
    differences = transformer_voltage_vs_usual(new_hour_phase, usual_voltage, yes_no([False, False]))
    # Check: A gets -5 V; B is empty.
    assert differences.iloc[0] == pytest.approx(-5.0)
    assert list(differences.isna()) == [False, True]


def test_transformer_voltage_vs_usual_is_empty_for_a_phase_the_rules_untrusted():
    """A phase row the rules untrusted gets no value. Here phase B is out (0 V): without this, it would read
    0 - 231 = -231 V, a huge fake sag."""
    # Prepare: A is normal (226 V); B reads 0 V and the outage rule untrusted its row. Both usually read 231 V.
    new_hour_phase = phase_table([
        phase_row(FEATURES_HOUR, "A", energy=40.0, homes_kwh=39.0, voltage=226.0),
        phase_row(FEATURES_HOUR, "B", energy=0.0, homes_kwh=0.0, voltage=0.0),
    ])
    usual_voltage = pd.Series([231.0, 231.0])
    # Run.
    differences = transformer_voltage_vs_usual(new_hour_phase, usual_voltage, yes_no([False, True]))
    # Check: A still gets -5 V; B is empty.
    assert differences.iloc[0] == pytest.approx(-5.0)
    assert list(differences.isna()) == [False, True]


# =============================================================================
# GAP STAGE (voltage_gap_with_neighbours.py): each home's voltage gap with its neighbours
# =============================================================================
def home_voltage_row(home_id, cable, phase, voltage, estimated=0):
    """One home row of the new hour, for the gap tests: the home, its cable and phase, its voltage and its estimated
    flag are what the gap reads (its kWh plays no part, so it's an ordinary 0.5)."""
    return home_row_at_features_hour(home_id, phase, kwh=0.5, estimated=estimated, cable=cable, voltage=voltage)


def cable_1_phase_a_homes():
    """Four neighbours: H1 to H4, all on cable 1, phase A, reading 226, 229, 228 and 230 V.

    A fresh list each time, so a test can add a home to it or change one.
    """
    return [
        home_voltage_row("H1", cable=1, phase="A", voltage=226.0),
        home_voltage_row("H2", cable=1, phase="A", voltage=229.0),
        home_voltage_row("H3", cable=1, phase="A", voltage=228.0),
        home_voltage_row("H4", cable=1, phase="A", voltage=230.0),
    ]


def run_voltage_gap_with_neighbours(rows, untrusted_homes=()):
    """Runs voltage_gap_with_neighbours on these home rows, with the rules untrusting the homes listed.

    untrusted_homes: e.g. {"H5"}; left out when the rules untrusted nobody.
    Returns one gap per row, in the rows' order.
    """
    new_hour_home = home_table(rows)
    is_home_untrusted_by_rules = new_hour_home["home_id"].isin(untrusted_homes)  # yes for each home listed
    return voltage_gap_with_neighbours(new_hour_home, is_home_untrusted_by_rules)


def test_voltage_gap_with_neighbours_is_the_gap_with_their_median():
    """Each home's gap: its voltage minus the median voltage of the other homes on its cable and phase."""
    # Prepare: H1 to H4 on cable 1, phase A, at 226, 229, 228 and 230 V. The rules untrusted nobody.
    # Run.
    gaps = run_voltage_gap_with_neighbours(cable_1_phase_a_homes())
    # Check, home by home (each has three neighbours, and the median of three values is the middle one):
    #   H1: 226 - median(229, 228, 230) = 226 - 229 = -3
    #   H2: 229 - median(226, 228, 230) = 229 - 228 = +1
    #   H3: 228 - median(226, 229, 230) = 228 - 229 = -1
    #   H4: 230 - median(226, 229, 228) = 230 - 228 = +2
    assert list(gaps) == pytest.approx([-3.0, 1.0, -1.0, 2.0])


def test_voltage_gap_with_neighbours_counts_only_the_same_cable_and_phase():
    """Homes on another cable, or on another phase of the same cable, aren't neighbours."""
    # Prepare: H1 to H4 as above, plus H5 on cable 2 (phase A) and H6 on phase B (cable 1), both at a wild 100 V.
    rows = cable_1_phase_a_homes()
    rows.append(home_voltage_row("H5", cable=2, phase="A", voltage=100.0))
    rows.append(home_voltage_row("H6", cable=1, phase="B", voltage=100.0))
    # Run.
    gaps = run_voltage_gap_with_neighbours(rows)
    # Check: H1 to H4 keep the gaps above. If H5 and H6 counted, H1's neighbours would be 229, 228, 230, 100 and 100,
    # whose median is 228, and H1's gap would be -2 instead of -3.
    assert list(gaps.iloc[:4]) == pytest.approx([-3.0, 1.0, -1.0, 2.0])  # iloc[:4]: the first four rows


def test_voltage_gap_with_neighbours_leaves_out_neighbours_the_rules_untrusted():
    """A neighbour whose reading the rules untrusted is left out of the median: its voltage may be broken."""
    # Prepare: H1 to H4 as above, plus H5 on cable 1, phase A, reading 0 V while sending kWh: the impossible rule
    # caught it, so the rules untrusted it.
    rows = cable_1_phase_a_homes()
    rows.append(home_voltage_row("H5", cable=1, phase="A", voltage=0.0))
    # Run.
    gaps = run_voltage_gap_with_neighbours(rows, untrusted_homes={"H5"})
    # Check: H1 to H4 keep the gaps above. If H5 counted, H1's neighbours would be 229, 228, 230 and 0, whose median
    # is 228.5 (the middle two, averaged), and H1's gap would be -2.5.
    assert list(gaps.iloc[:4]) == pytest.approx([-3.0, 1.0, -1.0, 2.0])


def test_voltage_gap_with_neighbours_still_counts_an_estimated_neighbour():
    """An estimated reading (the utility's guess) still counts as a neighbour: the rules don't untrust it, and one
    guessed voltage barely moves the median of 7 or 8 real neighbours."""
    # Prepare: H1 to H4 as above, but H4's 230 V is estimated.
    rows = cable_1_phase_a_homes()
    rows[3]["estimated"] = 1  # rows[3]: H4's row (the fourth; counting starts at 0)
    # Run.
    gaps = run_voltage_gap_with_neighbours(rows)
    # Check: H1 still gets -3. If H4 were left out, H1's neighbours would be 229 and 228, whose median is 228.5, and
    # H1's gap would be -2.5.
    assert gaps.iloc[0] == pytest.approx(-3.0)


def test_voltage_gap_with_neighbours_is_empty_without_a_usable_neighbour():
    """With no usable neighbour there's nothing to compare with, so the gap is empty."""
    # Prepare: H1 is alone on cable 2, phase A. H2 and H3 share cable 1, phase B, but the rules untrusted H3, so H2's
    # only neighbour can't be used.
    rows = [
        home_voltage_row("H1", cable=2, phase="A", voltage=226.0),
        home_voltage_row("H2", cable=1, phase="B", voltage=229.0),
        home_voltage_row("H3", cable=1, phase="B", voltage=0.0),
    ]
    # Run.
    gaps = run_voltage_gap_with_neighbours(rows, untrusted_homes={"H3"})
    # Check: H1 and H2 are empty (and H3 too, being untrusted itself: see the next test).
    assert list(gaps.isna()) == [True, True, True]


def test_voltage_gap_with_neighbours_is_empty_for_a_home_the_rules_untrusted():
    """A home whose reading the rules untrusted gets no gap: its voltage may be broken. Its neighbours still get
    theirs, computed without it."""
    # Prepare: H1 to H4 as above; the stuck rule caught H1, so the rules untrusted it.
    # Run.
    gaps = run_voltage_gap_with_neighbours(cable_1_phase_a_homes(), untrusted_homes={"H1"})
    # Check: H1 is empty. The others' neighbours no longer include H1's 226 V:
    #   H2: 229 - median(228, 230) = 229 - 229 = 0
    #   H3: 228 - median(229, 230) = 228 - 229.5 = -1.5
    #   H4: 230 - median(229, 228) = 230 - 228.5 = +1.5
    assert list(gaps.isna()) == [True, False, False, False]
    assert list(gaps.iloc[1:]) == pytest.approx([0.0, -1.5, 1.5])  # iloc[1:]: every row after the first


# =============================================================================
# FOREST FEATURES (forest_table.py): voltage_gap_with_neighbours_vs_usual
# =============================================================================
def test_voltage_gap_with_neighbours_vs_usual_subtracts_the_usual_gap():
    """The feature's basic job: the home's gap now minus its usual gap, in volts."""
    # Prepare: H1's gap is usually -3 V (it sits far down its cable) and is -3 V now. H2's gap is usually -3 V too,
    # but its meter's voltage sensor drifted: its gap is now -8 V.
    new_hour_home = home_table([home_row_at_features_hour("H1", "A", 0.5), home_row_at_features_hour("H2", "B", 0.5)])
    new_hour_home["voltage_gap_with_neighbours"] = [-3.0, -8.0]  # the gaps, as the gap stage writes them
    usual_gap = pd.Series([-3.0, -3.0])  # one per home row, as voltage_gap_with_neighbours_usuals would give them
    # Run.
    differences = voltage_gap_with_neighbours_vs_usual(new_hour_home, usual_gap)
    # Check: H1 is in line as usual (0 V); H2 is 5 V out of line with its neighbours (-5 V).
    assert list(differences) == pytest.approx([0.0, -5.0])


def test_voltage_gap_with_neighbours_vs_usual_is_empty_without_a_gap_or_a_usual():
    """No gap (the home, or all its neighbours, couldn't be used) or no usual gap yet: nothing to compare, so the
    feature is empty."""
    # Prepare: H1 has no gap; H2 has no usual gap yet; H3 has both (-8 V now, usually -3 V).
    new_hour_home = home_table([
        home_row_at_features_hour("H1", "A", 0.5),
        home_row_at_features_hour("H2", "B", 0.5),
        home_row_at_features_hour("H3", "C", 0.5),
    ])
    new_hour_home["voltage_gap_with_neighbours"] = [np.nan, -8.0, -8.0]
    usual_gap = pd.Series([-3.0, np.nan, -3.0])
    # Run.
    differences = voltage_gap_with_neighbours_vs_usual(new_hour_home, usual_gap)
    # Check: H1 and H2 are empty; H3 still gets -8 - (-3) = -5 V.
    assert list(differences.isna()) == [True, True, False]
    assert differences.iloc[2] == pytest.approx(-5.0)


# =============================================================================
# FOREST FEATURES (forest_table.py): build_forest_table
# =============================================================================
# A small street for the table tests: H1 and H2 on cable 1, phase A; H3 and H4 on cable 1, phase B. Its memory holds
# three earlier weekday evenings at 18:00, all trusted, so every usual exists at the new hour (Friday 18:00):
#   every home used 0.6 kWh each evening, so each home's usual consumption is 0.6 kWh;
#   each home's gap was the same every evening, so its usual gap is that value (USUAL_GAPS below);
#   each phase's transformer read 230, 231 and 232 V, so each phase's usual voltage is 231 V.
PAST_EVENINGS = [
    (pd.Timestamp(2019, 1, 15, 18), 230.0),  # Tuesday, and the transformer's voltage that evening (both phases)
    (pd.Timestamp(2019, 1, 16, 18), 231.0),  # Wednesday
    (pd.Timestamp(2019, 1, 17, 18), 232.0),  # Thursday
]
USUAL_GAPS = {"H1": -3.0, "H2": 3.0, "H3": -2.0, "H4": 2.0}


def forest_table_for_small_street(untrusted_homes=()):
    """Runs the gap stage, then build_forest_table, on the small street at the new hour, as check_hour will.

    The new hour:
      H1 used 0.9 kWh at 226 V, H2 0.3 kWh at 229 V (phase A); H3 0.6 kWh at 226 V, H4 1.2 kWh at 230 V (phase B);
      phase A: the transformer sent 40 kWh at 226 V, and the homes reported 37.98 kWh;
      phase B: the transformer sent 20 kWh at 231 V, and the homes reported 19.19 kWh.
    untrusted_homes: the homes the rules untrusted at the new hour, e.g. {"H1"}; left out when they untrusted nobody.
    Returns build_forest_table's two results: the forest table and the updated last usuals.
    """
    new_hour_home = home_table([
        home_row_at_features_hour("H1", "A", 0.9, voltage=226.0),
        home_row_at_features_hour("H2", "A", 0.3, voltage=229.0),
        home_row_at_features_hour("H3", "B", 0.6, voltage=226.0),
        home_row_at_features_hour("H4", "B", 1.2, voltage=230.0),
    ])
    new_hour_phase = phase_table([
        phase_row(FEATURES_HOUR, "A", energy=40.0, homes_kwh=37.98, voltage=226.0),
        phase_row(FEATURES_HOUR, "B", energy=20.0, homes_kwh=19.19, voltage=231.0),
    ])
    # The rules' answers: yes for each home listed; no phase row untrusted.
    is_home_untrusted_by_rules = new_hour_home["home_id"].isin(untrusted_homes)
    is_phase_untrusted_by_rules = yes_no([False, False])
    # The memory: the three trusted evenings, for every home and both phases.
    home_rows = []
    phase_rows = []
    for evening, transformer_voltage in PAST_EVENINGS:
        for home_id, usual_gap in USUAL_GAPS.items():
            home_rows.append(home_memory_row(evening, home_id, 0.6, voltage_gap_with_neighbours=usual_gap))
        phase_rows.append(phase_memory_row(evening, "A", transformer_voltage))
        phase_rows.append(phase_memory_row(evening, "B", transformer_voltage))
    # The gap stage, as check_hour will run it: each home's gap with its neighbours, written into its row.
    new_hour_home["voltage_gap_with_neighbours"] = voltage_gap_with_neighbours(
        new_hour_home, is_home_untrusted_by_rules
    )
    # The code being tested.
    return build_forest_table(
        new_hour_home,
        new_hour_phase,
        home_memory(home_rows),
        pd.DataFrame(phase_rows, columns=PHASE_MEMORY_COLUMNS),
        last_usuals_table([]),  # nothing remembered yet
        is_home_untrusted_by_rules,
        is_phase_untrusted_by_rules,
        fit_table_from_true_losses(),  # the losses fit, written out from TRUE_LOSSES
    )


def test_build_forest_table_has_one_row_per_home_and_the_agreed_columns():
    """One row per home, in the new hour's order, with the agreed columns: the name tag, then the five features."""
    # Run: the small street; the rules untrusted nobody.
    forest_table, _last_usuals = forest_table_for_small_street()
    # Check: the columns, written out here rather than imported, so a change to the agreed format is noticed.
    assert list(forest_table.columns) == [
        "timestamp", "home_id",  # the name tag: whose row it is
        "consumption_vs_usual", "voltage_gap_with_neighbours_vs_usual", "phase_unexplained_energy",
        "transformer_voltage_vs_usual", "hour",  # the five features, in the plan's order
    ]
    # One row per home, each tagged with the new hour; the hour feature is 18 (18:00).
    assert list(forest_table["home_id"]) == ["H1", "H2", "H3", "H4"]
    assert list(forest_table["timestamp"]) == [FEATURES_HOUR] * 4
    assert list(forest_table["hour"]) == [18, 18, 18, 18]


def test_build_forest_table_puts_each_homes_features_on_its_row():
    """Each home gets its own two home features, and its own phase's two phase features."""
    # Run.
    forest_table, _last_usuals = forest_table_for_small_street()
    # Check, home by home (H1 and H2 on phase A; H3 and H4 on phase B):
    #   consumption_vs_usual: 0.9 ÷ 0.6 = 1.5; 0.3 ÷ 0.6 = 0.5; 0.6 ÷ 0.6 = 1.0; 1.2 ÷ 0.6 = 2.0
    #   gap now (each home has one neighbour): H1 226 - 229 = -3; H2 229 - 226 = +3; H3 226 - 230 = -4;
    #     H4 230 - 226 = +4. Minus the usual gaps (-3, +3, -2, +2): 0, 0, -2, +2
    #   phase_unexplained_energy: A's expected losses are 0.02 + 0.001 x 40² = 1.62 kWh, so
    #     (40 - 37.98 - 1.62) ÷ 40 = 1%; B's are 0.01 + 0.002 x 20² = 0.81 kWh, so (20 - 19.19 - 0.81) ÷ 20 = 0%
    #   transformer_voltage_vs_usual: A 226 - 231 = -5 V; B 231 - 231 = 0 V
    assert list(forest_table["consumption_vs_usual"]) == pytest.approx([1.5, 0.5, 1.0, 2.0])
    assert list(forest_table["voltage_gap_with_neighbours_vs_usual"]) == pytest.approx([0.0, 0.0, -2.0, 2.0])
    assert list(forest_table["phase_unexplained_energy"]) == pytest.approx([1.0, 1.0, 0.0, 0.0])
    assert list(forest_table["transformer_voltage_vs_usual"]) == pytest.approx([-5.0, -5.0, 0.0, 0.0])


def test_build_forest_table_remembers_all_three_usuals():
    """Every usual computed for the new hour is remembered in the last usuals, each under its own name."""
    # Run.
    _forest_table, last_usuals = forest_table_for_small_street()
    # Check one of each: H1's consumption (0.6 kWh), H3's voltage gap (-2 V), phase A's transformer voltage (231 V).
    remembered_consumption = remembered_usuals(last_usuals, "H1", usual_of=settings.USUAL_OF_CONSUMPTION)
    remembered_gap = remembered_usuals(last_usuals, "H3", usual_of=settings.USUAL_OF_VOLTAGE_GAP_WITH_NEIGHBOURS)
    remembered_voltage = remembered_usuals(last_usuals, "A", usual_of=settings.USUAL_OF_TRANSFORMER_VOLTAGE)
    assert remembered_consumption == pytest.approx([0.6])
    assert remembered_gap == pytest.approx([-2.0])
    assert remembered_voltage == pytest.approx([231.0])


def test_build_forest_table_leaves_empty_what_a_caught_home_breaks():
    """A home the rules caught gets no home features, and empties what its reading is part of: its phase's
    unexplained energy (for every home on that phase), and its neighbour's gap when it was that neighbour's only one.
    The transformer's voltage and the hour don't depend on it, so they stay."""
    # Run: the stuck rule caught H1 (on phase A), so the rules untrusted it.
    forest_table, _last_usuals = forest_table_for_small_street(untrusted_homes={"H1"})
    # Check, column by column (rows: H1, H2, H3, H4):
    #   consumption: only H1 empty (H2 still gets its own 0.5)
    #   gap: H1 empty (its reading is broken) and H2 empty (H1 was its only neighbour)
    #   unexplained energy: H1 and H2 empty (H1's kWh is part of phase A's sum); phase B is untouched
    #   transformer voltage: unchanged (-5, -5, 0, 0)
    assert list(forest_table["consumption_vs_usual"].isna()) == [True, False, False, False]
    assert list(forest_table["voltage_gap_with_neighbours_vs_usual"].isna()) == [True, True, False, False]
    assert list(forest_table["phase_unexplained_energy"].isna()) == [True, True, False, False]
    assert list(forest_table["transformer_voltage_vs_usual"]) == pytest.approx([-5.0, -5.0, 0.0, 0.0])
