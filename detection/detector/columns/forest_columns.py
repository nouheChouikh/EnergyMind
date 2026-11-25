"""The five columns for one hour: what the forest looks at for every home (plan step 4).

What it does: turns the new hour's readings into the forest's columns. Three of the five are built so far:
  - consumption_vs_usual: each home's kWh ÷ its usual kWh (1.0 = as usual);
  - phase_unexplained_energy: the % of each phase's energy that neither its homes nor the expected cable losses
    explain (0% = everything adds up), only where is_balance_computable says the balance can be trusted;
  - transformer_voltage_vs_usual: each phase's transformer voltage − its usual voltage (0 V = as usual).
Still to come: voltage_gap_with_neighbours_vs_usual, hour, and the table that puts all five side by side on each
home's row (a home row gets its own phase's two phase columns).

Why a model and not if-else rules: each column alone is ambiguous. A home using less can be a family
on holiday or a theft; a home's voltage dropping can be the grid or a broken meter. Only the columns
together tell them apart (less consumption + energy missing on the phase = suspicious; less consumption
+ energy adding up = normal life). The forest learns from a year of normal hours which combinations are
normal, including combinations nobody wrote a rule for. An if-else rule for every combination of five
columns can't be written; the forest scores the whole row at once.

Choices:
  - A column is empty (NaN) when one of its inputs is missing or untrusted by the rules. An empty column makes the
    row unchecked (not scored), which is honest; a wrong value would look real to the forest.
  - Each function gets the new hour's rows plus values already computed elsewhere (the usuals, the expected losses,
    the rules' answers), so each one can be checked on its own with small made-up numbers.
  - Three forms, each suited to its value:
      consumption is a ratio: homes differ in size (0.3 kWh more is a lot for a small home, little for a big one);
      voltage is a difference: every home sits near 230 V, so volts already mean the same everywhere;
      the balance is a % of the phase's energy: the transformer meter's error is a % of what it measures, so the
      normal range stays the same at every load.
  - A home the rules caught empties its whole phase's balance: its kWh is part of the phase's sum, so the sum is
    wrong. An estimated reading does the same: it's the utility's guess, and a single hour can't average a guess
    out (the losses fit, over a whole year, can).
"""

import numpy as np  # np.nan: an empty value

from detector import settings  # the minimum under the consumption usual


# =============================================================================
# HOME COLUMNS: one value per home row
# =============================================================================
def consumption_vs_usual(new_hour_home, usual_kwh, is_home_untrusted):
    """Each home's kWh at the new hour divided by its usual kWh: 1.0 = as usual, 1.5 = 50% more, 0.2 = 80% less.

    new_hour_home: the home table's rows for the hour being checked (one row per home).
    usual_kwh: each home row's usual kWh, as consumption_usuals returns it (empty for a home with no usual yet).
    is_home_untrusted: one yes/no per home row, as is_home_untrusted_by_rules returns it.
    Returns one number per home row, in the same order. Empty when the home sent no kWh, has no usual yet, or was
    untrusted by the rules (its reading is broken, or an outage or a collection failure covers it).
    """
    # A usual below the minimum counts as the minimum, so a tiny night usual can't make the ratio jump
    # (0.08 kWh against a usual of 0.01 would read "8 times usual"), and a usual of 0 can't divide by zero.
    # clip(lower=...) raises every value below the minimum up to it; the others, and the empty ones, stay as they are.
    usual_used = usual_kwh.clip(lower=settings.CONSUMPTION_USUAL_MINIMUM_KWH)
    # Row by row: this hour's kWh ÷ the usual. Empty wherever either one is empty.
    ratios = new_hour_home["kwh"] / usual_used
    # A reading the rules untrusted gets no value: it's broken, and must never look real to the forest.
    ratios.loc[is_home_untrusted] = np.nan
    return ratios


# =============================================================================
# PHASE COLUMNS: one value per phase row
# =============================================================================
def is_balance_computable(new_hour_phase, new_hour_home, is_home_untrusted, is_phase_untrusted):
    """For each phase row: can its energy balance be trusted at the new hour? One yes/no per phase row.

    The balance compares what the transformer sent on the phase with what the phase's homes reported, so it only
    means something if every number in it is a real, trusted measurement. A phase says yes only when:
      1. the transformer measured energy on it (above 0: the balance is a % of that energy, so it divides by it);
      2. the rules didn't untrust its row (its transformer reading isn't broken, and it isn't out);
      3. the rules untrusted none of its homes (a caught reading is part of the phase's sum, so the sum is wrong;
         a missing reading is always caught by the missing rule, so this covers it too);
      4. none of its homes sent an estimated reading (the utility's guess, not a measurement).
    new_hour_phase / new_hour_home: the phase table's and the home table's rows for the hour being checked.
    is_home_untrusted / is_phase_untrusted: the rules' answers, as is_home_untrusted_by_rules and
    is_phase_untrusted_by_rules return them.
    """
    # 1. Energy measured. An empty reading isn't "above 0", so it counts as not measured.
    is_energy_measured = new_hour_phase["transformer_kwh"] > 0
    # 3 and 4. A home whose kWh can't go into its phase's sum: untrusted by the rules, or estimated.
    is_estimated = new_hour_home["estimated"] == 1
    is_home_kwh_unusable = is_home_untrusted | is_estimated  # "|" means "or"
    phases_with_unusable_kwh = new_hour_home.loc[is_home_kwh_unusable, "phase"]  # e.g. ["B"]
    has_unusable_kwh = new_hour_phase["phase"].isin(phases_with_unusable_kwh)  # yes for each of those phases
    # All four must hold. "~" means "not": measured, and not untrusted (2), and no unusable kWh (3 and 4).
    return is_energy_measured & ~is_phase_untrusted & ~has_unusable_kwh


def phase_unexplained_energy(new_hour_phase, expected_losses_kwh, is_computable):
    """The share of each phase's energy that neither its homes nor the expected cable losses explain, in %.

    (transformer energy − sum of its homes' kWh − expected losses) ÷ transformer energy × 100:
      0%: everything adds up;
      above 0: energy is missing (it left the transformer, but no meter recorded it: e.g. a theft);
      below 0: energy is appearing (the meters reported more than the transformer sent).
    new_hour_phase: the phase table's rows for the hour being checked (one row per phase).
    expected_losses_kwh: each phase row's expected cable losses, as expected_losses returns them.
    is_computable: one yes/no per phase row, as is_balance_computable returns it.
    Returns one % per phase row, in the same order; empty where the balance can't be computed.
    """
    energy = new_hour_phase["transformer_kwh"]  # what the transformer sent on each phase
    # What's left once the homes' meters and the cables' expected losses are taken away.
    unexplained_kwh = energy - new_hour_phase["homes_kwh"] - expected_losses_kwh
    # As a % of what was sent, so the normal range is the same at every load.
    percentages = unexplained_kwh / energy * 100
    # A balance built on a wrong or missing number gets no value.
    percentages.loc[~is_computable] = np.nan
    return percentages


def transformer_voltage_vs_usual(new_hour_phase, usual_voltage, is_phase_untrusted):
    """Each phase's transformer voltage at the new hour minus its usual voltage, in volts: 0 = as usual.

    A sag coming from the grid reads below 0, a surge above 0.
    new_hour_phase: the phase table's rows for the hour being checked (one row per phase).
    usual_voltage: each phase row's usual voltage, as transformer_voltage_usuals returns it (empty if none yet).
    is_phase_untrusted: one yes/no per phase row, as is_phase_untrusted_by_rules returns it.
    Returns one number per phase row, in the same order. Empty when the phase has no usual yet, or when the rules
    untrusted its row: its transformer reading is broken, or the phase is out (0 V would read as a huge fake sag).
    """
    # Row by row: this hour's voltage − the usual. Empty wherever the usual is empty.
    differences = new_hour_phase["transformer_voltage_v"] - usual_voltage
    # A reading the rules untrusted gets no value.
    differences.loc[is_phase_untrusted] = np.nan
    return differences
