"""Expected cable losses on each phase (plan step 3).

What it does:
  - fit_losses_regression: once, at learning, finds a and b of expected losses = a + b x E² for each phase.
    E is the transformer's energy on the phase; the losses are E minus the sum of the phase's meters.
  - expected_losses: live, every hour, gives each phase's expected losses at the new hour.

Why it exists: the energy balance (phase_unexplained_energy, plan step 4) must know how much energy the cables
normally lose. Losses are the only part of the balance nobody measures, and they grow with the square of the load,
so a fixed allowance can't fit both quiet and busy hours.

Choices:
  - The fit learns only from clean hours: the transformer measured energy, every home on the phase sent its kWh,
    and no rule finding touches that phase at that hour (a home's or the transformer's finding touches its phase;
    a street finding touches all three). One broken reading can outweigh weeks of real losses, and a least-squares
    line is pulled hardest by the biggest errors.
  - It's a linear regression on E² (scikit-learn's LinearRegression): a straight line drawn against E² instead of
    E. Only its two numbers, a and b, are kept, so the saved model stays plain numbers.
  - The fit stops with an error naming the phase when b isn't positive (losses that don't grow with the load
    aren't physical) or when a phase has no usable hour (e.g. one of its meters never reported).
  - No energy sent, nothing lost: expected losses are 0 when E is 0. An empty or negative E is a broken reading
    (the transformer rule catches it), so it has no expected losses: the answer is empty.
"""

import pandas as pd  # tables
from sklearn.linear_model import LinearRegression  # draws the best straight line through points (least squares)

from detector import settings  # the three phases
from detector.data_formats import LOSSES_FIT_COLUMNS  # the fit's columns: phase, a, b


# =============================================================================
# THE FIT (once, at learning)
# =============================================================================
def fit_losses_regression(phase_hours, rule_findings):
    """a and b of expected losses = a + b x E², for each phase, learned from the clean hours only.

    phase_hours: the phase table's rows for the learning hours (one row per phase and hour).
    rule_findings: the rules' findings for those same hours (the table rules/findings.py writes).
    Returns a table with the columns of LOSSES_FIT_COLUMNS: one row per phase, A then B then C.
    Raises ValueError, naming the phase, if a phase has no usable hour or its b isn't positive.
    """
    usable = usable_hours(phase_hours, rule_findings)
    rows = []  # one row (a dictionary) per phase, turned into a table at the end
    for phase in settings.PHASES:
        one_phase = usable[usable["phase"] == phase]
        if one_phase.empty:
            raise ValueError(
                f"phase {phase} has no usable hour to fit its losses (no complete hour that no rule caught): "
                "a meter on it may never have reported"
            )
        energy = one_phase["transformer_kwh"]  # E: what the transformer sent on this phase, each hour
        losses = energy - one_phase["homes_kwh"]  # what the meters didn't receive: lost in the cables
        # The straight line against E², the regression's only input column: losses = a + b x E².
        line = LinearRegression().fit((energy**2).to_frame("energy_squared"), losses)
        a = float(line.intercept_)  # the line's height where E² = 0: the fixed part
        b = float(line.coef_[0])  # the line's slope: how fast the losses grow with E²
        if b <= 0:
            raise ValueError(f"phase {phase}: its losses don't grow with the load (b = {b:.3g}), which isn't physical")
        rows.append({"phase": phase, "a": a, "b": b})
    return pd.DataFrame(rows, columns=LOSSES_FIT_COLUMNS)


# =============================================================================
# WHICH HOURS THE FIT LEARNS FROM
# =============================================================================
def usable_hours(phase_hours, rule_findings):
    """The phase table's rows the fit may learn from: complete hours that no rule finding touches."""
    # Complete: the transformer measured energy, and every home on the phase sent its kWh.
    energy_measured = phase_hours["transformer_kwh"] > 0  # an empty value fails "> 0" too
    all_homes_sent = phase_hours["nb_homes_reporting"] == phase_hours["nb_homes_on_phase"]
    # Untouched: no rule finding concerns this phase at this hour.
    touched = touched_phase_hours(rule_findings)
    is_touched = pd.Series(
        [(hour, phase) in touched for hour, phase in zip(phase_hours["timestamp"], phase_hours["phase"])],
        index=phase_hours.index,
    )
    return phase_hours[energy_measured & all_homes_sent & ~is_touched]


def touched_phase_hours(rule_findings):
    """Every (hour, phase) pair a rule finding touches, as a set.

    A finding with a phase touches that phase: a home's finding carries the home's phase, a transformer finding the
    broken phase, an outage the phase that's out. A street finding has no phase: it touches all three phases.
    """
    has_phase = rule_findings["phase"].notna()
    touched = set(zip(rule_findings.loc[has_phase, "timestamp"], rule_findings.loc[has_phase, "phase"]))
    for hour in rule_findings.loc[~has_phase, "timestamp"]:  # the street findings' hours
        touched |= {(hour, phase) for phase in settings.PHASES}
    return touched


# =============================================================================
# EXPECTED LOSSES (live, every hour)
# =============================================================================
def expected_losses(new_hour_phase, losses_fit):
    """Each phase's expected cable losses at the new hour, in kWh.

    new_hour_phase: the phase table's rows for the hour being checked (one row per phase).
    losses_fit: the fit's table (one row per phase: phase, a, b), as learning returned it.
    Returns one value per row of new_hour_phase, in the same order:
      - a + b x E² when the transformer sent energy (E above 0);
      - 0 when E is 0: no energy sent, nothing lost;
      - empty when E is empty or negative: a broken reading has no expected losses.
    """
    energy = new_hour_phase["transformer_kwh"]
    # Each row's own a and b, looked up by its phase.
    fit_by_phase = losses_fit.set_index("phase")
    a = new_hour_phase["phase"].map(fit_by_phase["a"])
    b = new_hour_phase["phase"].map(fit_by_phase["b"])
    # The formula where energy was sent; empty everywhere else (E of 0, empty or negative).
    losses = (a + b * energy**2).where(energy > 0)
    # No energy sent, nothing lost: 0, not the fit's fixed part a.
    return losses.mask(energy == 0, 0.0)
