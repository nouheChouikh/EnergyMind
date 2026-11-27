"""Trust: which home rows and phase rows may feed "usual" (plan step 4, the trust rule).

What it does, in three parts:
  - the rules' answers: for each home row and phase row of the new hour, did the rules make it untrusted?
      a home's own finding (missing, impossible, stuck) untrusts that home;
      an outage on a phase untrusts every home on it and its phase row; a street outage untrusts everything;
      all meters missing untrusts every home, but no phase row (the transformer's reading arrived);
      a transformer problem on a phase untrusts that phase row only (the homes' own readings are fine).
  - which home rows the forest can score: those the rules didn't untrust, that were measured (not estimated), and
    that have all five features.
  - the trust states during learning: each row's trust state (trusted, untrusted or unchecked) while there's no
    forest yet, decided by the plan's learning rule.

Why it exists: only trusted rows may feed "usual". A reading the rules caught must never feed one, a row nobody
could check stays out too, and the forest only learns from (and later only scores) the rows it can score.

Choices:
  - The rules' answers are True or False: the rules can only say "untrusted" or "not decided yet". The trust states
    come once everything else is known; the caller writes them into the rows, and they go into the memory.
  - The rules' answers read only the findings table, so no rule logic is repeated here. Homes covered by an outage
    have no finding row of their own, which is why the outage's phase (or the street scope) is read as well.
  - A function's answer keeps the function's name when it's passed on: in the functions below,
    is_home_untrusted_by_rules is the yes/no that the function is_home_untrusted_by_rules returned.
  - Two kinds of trust states. During learning (built now) there's no forest yet, since learning is what makes it:
    a row the forest could score is trusted, because the learning year is assumed mostly normal. Live (built with
    the forest), the forest and the diagnosis will decide those rows. Everything else is the same in both: a row the
    rules untrusted is untrusted; an estimated reading is trusted (the utility's own value, kept for history, never
    scored); a row with an empty feature is unchecked (nobody could check it).
  - A phase row only feeds the transformer voltage usual, so only its own reading decides its trust state. Its
    balance plays no part: a caught home meter empties the balance, but says nothing about the transformer's voltage.
  - It sits at the top of detector/, not in features/: it isn't something a feature is built from. It decides what
    the memory keeps.
  - Built for live use: everything here judges the new hour only. Operator decisions live in
    feedback/operator_decisions.py.
"""

import numpy as np  # np.select: for each row, the trust state of the first case that holds
import pandas as pd  # tables

from detector import settings  # the scope, verdict and trust state names
from detector.data_formats import FOREST_FEATURES  # the five features


# =============================================================================
# THE RULES' ANSWERS: what the rules make untrusted at the new hour
# =============================================================================
def is_home_untrusted_by_rules(new_hour_home, rule_findings):
    """One True/False per home row of the new hour: True if the rules make that home's reading untrusted.

    new_hour_home: the home table's rows for the hour being checked (one row per home).
    rule_findings: that hour's findings table, as rules/findings.py writes it.
    Returns the answers in the same order as new_hour_home's rows.
    """
    scope = rule_findings["scope"]  # where each finding sits: home, phase, street or transformer

    # A home with a finding of its own (missing, impossible or stuck).
    homes_with_own_finding = rule_findings.loc[scope == settings.SCOPE_HOME, "home_id"]  # their home ids
    has_own_finding = new_hour_home["home_id"].isin(homes_with_own_finding)

    # A home on a phase that's out: the outage covers it (it has no finding row of its own).
    phases_out = rule_findings.loc[scope == settings.SCOPE_PHASE, "phase"]  # the phases that are out
    is_on_a_phase_out = new_hour_home["phase"].isin(phases_out)

    # A street finding (a street outage, or all meters missing) covers every home.
    has_street_finding = (scope == settings.SCOPE_STREET).any()  # a single True/False for the whole hour

    return has_own_finding | is_on_a_phase_out | has_street_finding  # "|" means "or"


def is_phase_untrusted_by_rules(new_hour_phase, rule_findings):
    """One True/False per phase row of the new hour: True if the rules make that phase row untrusted.

    new_hour_phase: the phase table's rows for the hour being checked (one row per phase).
    rule_findings: that hour's findings table, as rules/findings.py writes it.
    Returns the answers in the same order as new_hour_phase's rows.
    """
    scope = rule_findings["scope"]  # where each finding sits: home, phase, street or transformer

    # A phase whose transformer reading is broken, or that's out.
    is_broken_or_out_finding = scope.isin([settings.SCOPE_TRANSFORMER, settings.SCOPE_PHASE])
    phases_broken_or_out = rule_findings.loc[is_broken_or_out_finding, "phase"]  # those phases' names
    is_broken_or_out = new_hour_phase["phase"].isin(phases_broken_or_out)

    # A street outage puts all three phases out. All meters missing doesn't: the transformer's reading arrived.
    has_street_outage = (rule_findings["verdict"] == settings.VERDICT_OUTAGE_STREET).any()  # one True/False

    return is_broken_or_out | has_street_outage  # "|" means "or"


# =============================================================================
# WHICH HOME ROWS THE FOREST CAN SCORE
# =============================================================================
def is_home_row_scorable(new_hour_home, is_home_untrusted_by_rules, forest_table):
    """For each home row of the new hour: can the forest score it? One yes/no per home row.

    Yes only when all three hold:
      1. the rules didn't untrust it (a caught reading is broken);
      2. its reading was measured, not estimated (an estimated reading is the utility's guess: kept, never scored);
      3. all five of its features exist (with an empty feature, there's nothing complete to score).
    During learning, these are the rows the forest learns from (from 29 January on); live, the rows it scores.
    new_hour_home: the home table's rows for the hour being checked (one row per home).
    is_home_untrusted_by_rules: one yes/no per home row, as the function is_home_untrusted_by_rules returns it.
    forest_table: the forest table for that hour, as build_forest_table returns it (the same homes, in the same order).
    """
    is_estimated = new_hour_home["estimated"] == 1
    # notna: "is this value filled in?"; all(axis=1): are all five filled in, row by row?
    has_every_feature = forest_table[FOREST_FEATURES].notna().all(axis=1)
    return ~is_home_untrusted_by_rules & ~is_estimated & has_every_feature  # "~" means "not", "&" means "and"


# =============================================================================
# TRUST STATES DURING LEARNING: there's no forest yet
# =============================================================================
def home_trust_states_during_learning(new_hour_home, is_home_untrusted_by_rules, forest_table):
    """Each home row's trust state at the new hour, during learning: trusted, untrusted or unchecked.

    There's no forest yet (learning is what makes it), so the plan's learning rule decides: a row the forest could
    score is trusted, because the learning year is assumed mostly normal.
    new_hour_home, is_home_untrusted_by_rules, forest_table: as for is_home_row_scorable.
    Returns one trust state per home row, in the same order (their names are in settings.py). The caller writes them
    into the home rows (column trust), so they go into the memory.
    """
    is_estimated = new_hour_home["estimated"] == 1
    is_scorable = is_home_row_scorable(new_hour_home, is_home_untrusted_by_rules, forest_table)
    # np.select gives each row the trust state of the first case that holds:
    #   1. the rules untrusted it          → untrusted
    #   2. its reading is estimated        → trusted (the utility's own value: kept for history, never scored)
    #   3. the forest could score it       → trusted (the learning rule)
    #   otherwise (one feature is empty)   → unchecked (nobody could check it)
    trust_states = np.select(
        [is_home_untrusted_by_rules, is_estimated, is_scorable],
        [settings.TRUST_UNTRUSTED, settings.TRUST_TRUSTED, settings.TRUST_TRUSTED],
        default=settings.TRUST_UNCHECKED,
    )
    return pd.Series(trust_states, index=new_hour_home.index)  # lined up with the home rows


def phase_trust_states_during_learning(new_hour_phase, new_hour_home, is_phase_untrusted_by_rules, forest_table):
    """Each phase row's trust state at the new hour, during learning: trusted, untrusted or unchecked.

    A phase row only feeds the transformer voltage usual, so only its own reading decides.
    new_hour_phase: the phase table's rows for the hour being checked (one row per phase).
    new_hour_home: the home table's rows for that hour: they say which phase each forest table row is on.
    is_phase_untrusted_by_rules: one yes/no per phase row, as the function is_phase_untrusted_by_rules returns it.
    forest_table: the forest table for that hour, as build_forest_table returns it (the same homes as new_hour_home).
    Returns one trust state per phase row, in the same order. The caller writes them into the phase rows.
    """
    # The forest table has no phase rows, but every home row carries its own phase's transformer voltage feature.
    # A phase whose feature is empty on its homes' rows had no usual voltage to compare with.
    is_feature_empty_on_home_row = forest_table["transformer_voltage_vs_usual"].isna()  # one yes/no per home row
    phases_with_empty_feature = new_hour_home.loc[is_feature_empty_on_home_row, "phase"]  # e.g. ["B"]
    is_transformer_voltage_feature_empty = new_hour_phase["phase"].isin(phases_with_empty_feature)  # per phase row
    # np.select gives each row the trust state of the first case that holds:
    #   1. the rules untrusted it (its transformer reading is broken, or it's out)   → untrusted
    #   2. its transformer voltage feature is empty (no usual voltage yet)           → unchecked
    #   otherwise                                                                    → trusted (the learning rule)
    trust_states = np.select(
        [is_phase_untrusted_by_rules, is_transformer_voltage_feature_empty],
        [settings.TRUST_UNTRUSTED, settings.TRUST_UNCHECKED],
        default=settings.TRUST_TRUSTED,
    )
    return pd.Series(trust_states, index=new_hour_phase.index)  # lined up with the phase rows
