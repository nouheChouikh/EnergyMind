"""Expected cable losses on each phase (plan step 3).

What it does:
  - fit_losses_regression: once, at learning, finds a and b of expected losses = a + b x E² for each phase.
    E is the transformer's energy on the phase; the losses are E minus the sum of the phase's meters.
  - expected_losses: live, every hour, gives each phase's expected losses at the new hour.

Why it exists: phase_unexplained_energy (plan step 4) must know how much energy the cables normally lose. In it
(energy sent − energy the homes recorded − expected losses), losses are the only part nobody measures, and they
grow with the square of the load, so a fixed allowance can't fit both quiet and busy hours.

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

import numpy as np  # np.select: picks one answer per row from a list of cases; np.nan: an empty value
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
    usable = usable_hours(phase_hours, rule_findings)  # the rows the fit may learn from
    fit_rows = []  # one row (a dictionary) per phase, turned into a table at the end
    for phase in settings.PHASES:
        one_phase = usable[usable["phase"] == phase]  # this phase's usable rows
        if one_phase.empty:
            raise ValueError(
                f"phase {phase} has no usable hour to fit its losses (no complete hour that no rule caught): "
                "a meter on it may never have reported"
            )
        energy = one_phase["transformer_kwh"]  # E: what the transformer sent on this phase, each hour
        losses = energy - one_phase["homes_kwh"]  # what the meters didn't receive: lost in the cables
        # scikit-learn wants its input as a table, one column per input: here a single column, E².
        energy_squared = (energy**2).to_frame()
        # The straight line through the points (E², losses): losses = a + b x E².
        line = LinearRegression().fit(energy_squared, losses)
        a = float(line.intercept_)  # where the line starts (at E² = 0): the fixed part
        b = float(line.coef_[0])  # the line's slope: how fast the losses grow with E²
        if b <= 0:
            raise ValueError(f"phase {phase}: its losses don't grow with the load (b = {b:.3g}), which isn't physical")
        fit_rows.append({"phase": phase, "a": a, "b": b})
    return pd.DataFrame(fit_rows, columns=LOSSES_FIT_COLUMNS)


# =============================================================================
# WHICH HOURS THE FIT LEARNS FROM
# =============================================================================
def usable_hours(phase_hours, rule_findings):
    """The phase table's rows the fit may learn from: complete hours that no rule finding touches."""
    # Complete: the transformer measured energy, and every home on the phase sent its kWh.
    is_energy_measured = phase_hours["transformer_kwh"] > 0  # an empty value fails "> 0" too
    has_every_home_reported = phase_hours["nb_homes_reporting"] == phase_hours["nb_homes_on_phase"]
    is_complete = is_energy_measured & has_every_home_reported

    # Touched: go through the findings one at a time, and mark the rows each one concerns.
    is_touched = pd.Series(False, index=phase_hours.index)  # nothing marked yet
    for finding in rule_findings.itertuples():  # one finding at a time: finding.timestamp, finding.phase...
        is_same_hour = phase_hours["timestamp"] == finding.timestamp
        if pd.isna(finding.phase):
            # A street finding has no phase: it concerns all three phases at its hour.
            is_touched = is_touched | is_same_hour
        else:
            # A home's, the transformer's or an outage's finding concerns only its own phase.
            is_touched = is_touched | (is_same_hour & (phase_hours["phase"] == finding.phase))

    return phase_hours[is_complete & ~is_touched]  # "~" means "not": complete and not touched


# =============================================================================
# EXPECTED LOSSES (live, every hour)
# =============================================================================
def expected_losses(new_hour_phase, losses_fit):
    """Each phase's expected cable losses at the new hour, in kWh: one value per row of new_hour_phase, in order.

    new_hour_phase: the phase table's rows for the hour being checked (one row per phase).
    losses_fit: the fit's table (one row per phase: phase, a, b), as learning returned it.
    """
    energy = new_hour_phase["transformer_kwh"]  # E, for each phase row
    # Each row's own a and b, looked up in the fit by the row's phase.
    fit_by_phase = losses_fit.set_index("phase")  # the fit, findable by phase
    a = new_hour_phase["phase"].map(fit_by_phase["a"])
    b = new_hour_phase["phase"].map(fit_by_phase["b"])
    # One answer per row, by case:
    #   energy sent (E above 0)   ->  the formula, a + b x E²
    #   no energy sent (E is 0)   ->  0: nothing was lost
    #   empty or negative E       ->  empty: a broken reading has no expected losses
    answers = np.select([energy > 0, energy == 0], [a + b * energy**2, 0.0], default=np.nan)
    return pd.Series(answers, index=new_hour_phase.index)
