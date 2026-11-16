"""Findings: every definition rule at the new hour, in a fixed order, as one table (plan step 2).

What it does: runs the street rules, then the meter rules, and writes one row per finding for the new hour:
  1. transformer problems: one row per broken phase;
  2. outages: one street row if all 3 phases are out, otherwise one row per phase that's out;
  3. all meters missing: one street row;
  4. missing, impossible, stuck: one row per home, for the homes no street finding covers.

Why it exists: each rule answers yes or no on its own. The rest of the system needs one ordered answer per hour,
where one problem gives one finding (a street-wide outage is one row, not 75 home rows).

Choices:
  - Street findings come first and cover their homes for that hour: the homes on a phase that's out, and every
    home during "all meters missing" (readings that slip through a failing collection system can't be trusted).
    A covered home gets no row of its own.
  - A transformer problem covers no home: the homes' own readings are fine.
  - Each home gets at most one verdict: missing, then impossible, then stuck. A meter frozen on an impossible
    value is reported as impossible, the stronger statement.
  - Built for live use: it judges the new hour only, from the new hour and the memory.
"""

import numpy as np  # np.select: the first verdict that fires, for each home
import pandas as pd  # tables

from detector import settings  # the verdict and scope names
from detector.data_formats import RULE_FINDINGS_COLUMNS  # the findings table's columns
from detector.rules.meter_rules import find_impossible, find_missing, find_stuck  # the meter rules
from detector.rules.street_rules import find_missing_all, find_outages, find_transformer_problems  # street rules


# =============================================================================
# ONE ROW
# =============================================================================
def finding(hour, scope, verdict, phase=None, home_id=None):
    """One row of the findings table. phase and home_id stay empty when the finding doesn't concern one."""
    return {"timestamp": hour, "scope": scope, "phase": phase, "home_id": home_id, "verdict": verdict}


# =============================================================================
# ALL THE RULES, IN ORDER
# =============================================================================
def find_rule_findings(new_hour_home, new_hour_phase, home_memory, phase_memory):
    """Every rule finding at the new hour, one row each.

    new_hour_home: the home table's rows for the hour being checked (one row per home).
    new_hour_phase: the phase table's rows for that hour (one row per phase).
    home_memory, phase_memory: the two tables' rows for earlier hours, as kept by the operator app.
    Returns a table with the columns of RULE_FINDINGS_COLUMNS, one row per finding (empty if no rule fires).
    """
    hour = new_hour_home["timestamp"].iloc[0]  # the hour being checked
    findings = []  # one row (a dictionary) per finding, turned into a table at the end

    # ----------------------------------------------------------------------------- 1. transformer problems
    # One row per phase where the transformer's own meter is wrong. The homes' readings are fine: no home is covered.
    broken = find_transformer_problems(new_hour_home, new_hour_phase, phase_memory)
    for phase in new_hour_phase.loc[broken, "phase"]:
        findings.append(finding(hour, settings.SCOPE_TRANSFORMER, settings.VERDICT_REFERENCE_METER, phase=phase))

    # ----------------------------------------------------------------------------- 2. outages
    # All 3 phases out: one row for the whole street. Otherwise: one row per phase that's out.
    out = find_outages(new_hour_home, new_hour_phase)
    phases_out = list(new_hour_phase.loc[out, "phase"])
    if len(phases_out) == len(settings.PHASES):
        findings.append(finding(hour, settings.SCOPE_STREET, settings.VERDICT_OUTAGE_STREET))
    else:
        for phase in phases_out:
            findings.append(finding(hour, settings.SCOPE_PHASE, settings.VERDICT_OUTAGE_PHASE, phase=phase))

    # ----------------------------------------------------------------------------- 3. all meters missing
    # One row for the whole street: the data link failed.
    missing_all = find_missing_all(new_hour_home, new_hour_phase)
    if missing_all:
        findings.append(finding(hour, settings.SCOPE_STREET, settings.VERDICT_MISSING_ALL))

    # ----------------------------------------------------------------------------- 4. home findings
    # Which homes a street finding covers: every home during a data link failure; otherwise the homes on a phase
    # that's out. A covered home gets no row of its own.
    if missing_all:
        covered = pd.Series(True, index=new_hour_home.index)
    else:
        covered = new_hour_home["phase"].isin(phases_out)

    # Each home's verdict: the first meter rule that fires, in this order (missing, impossible, stuck);
    # "" when none fires.
    verdicts = np.select(
        [find_missing(new_hour_home), find_impossible(new_hour_home), find_stuck(new_hour_home, home_memory)],
        [settings.VERDICT_MISSING_ONE, settings.VERDICT_IMPOSSIBLE, settings.VERDICT_STUCK],
        default="",
    )
    home_verdicts = pd.Series(verdicts, index=new_hour_home.index)

    # One row per home that has a verdict and isn't covered.
    to_report = (home_verdicts != "") & ~covered
    reported = new_hour_home[to_report]
    for home_id, phase, verdict in zip(reported["home_id"], reported["phase"], home_verdicts[to_report]):
        findings.append(finding(hour, settings.SCOPE_HOME, verdict, phase=phase, home_id=home_id))

    return pd.DataFrame(findings, columns=RULE_FINDINGS_COLUMNS)
