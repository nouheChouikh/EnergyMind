"""Trust: which home rows and phase rows may feed "usual" (plan step 4, the trust rule). For now, the rules' part.

What it does: reads the new hour's rule findings and answers, for every home row and every phase row, whether the
rules make it untrusted:
  - a home's own finding (missing, impossible, stuck) untrusts that home;
  - an outage on a phase untrusts every home on it and its phase row; a street outage untrusts everything;
  - all meters missing untrusts every home, but no phase row (the transformer's reading arrived);
  - a transformer problem on a phase untrusts that phase row only (the homes' own readings are fine).

Why it exists: only trusted rows may feed "usual", so a reading the rules caught or covered must never feed one,
and it isn't scored by the forest.

Choices:
  - The answers are True or False, like the rules' answers. The rules can only say "untrusted" or "not decided
    yet": a row they didn't catch is decided later (unchecked if a column can't be computed, trusted if it's an
    estimated reading, otherwise the forest decides). The trust words are written once everything is known.
  - It reads only the findings table, so no rule logic is repeated here. Homes covered by an outage have no
    finding row of their own, which is why the outage's phase (or the street scope) is read as well.
  - Built for live use: it judges the new hour only, from that hour's findings.
  - Later steps add the rest of the trust decision to this file; operator decisions live in
    feedback/operator_decisions.py.
"""

from detector import settings  # the scope and verdict names a finding is written with


# =============================================================================
# HOME ROWS
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


# =============================================================================
# PHASE ROWS
# =============================================================================
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
