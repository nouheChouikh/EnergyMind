"""Every value chosen by people: limits, windows, names (the detector's only settings file).

Values learned from data never go here: they live in the trained model.
Each part of the detector adds its own values when it is built.
"""

# ----------------------------------------------------------------------------- the street
# The transformer's three phases, in the order used everywhere.
PHASES = ("A", "B", "C")

# ----------------------------------------------------------------------------- types of day (plan step 4)
# Holidays join weekends: 4 weeks rarely hold more than one or two holidays, too few for a median of their own.
DAY_TYPES = ("weekday", "weekend_or_holiday")

# ----------------------------------------------------------------------------- the usual (plan step 4)
# The usual is the median of the same hour, on the same type of day, over this many calendar days before the new
# hour, using trusted rows only. Calendar days keep the comparison in the current season.
USUAL_WINDOW_DAYS = 28
# With fewer trusted values than this, there's no median: the last usual remembered is used instead.
USUAL_MIN_VALUES = 3
# The usuals' names in the last usuals table (column usual_of, see data_formats.py).
USUAL_OF_CONSUMPTION = "consumption"  # a home's kWh
USUAL_OF_GRID_VOLTAGE = "grid_voltage"  # a phase's transformer voltage

# ----------------------------------------------------------------------------- definition rules (plan step 2)
# Stuck meter: the whole reading (kWh, V and I) identical this many hours in a row.
# Honest meters repeat their whole reading at most 3 hours in a row (dataset checks, 2018).
STUCK_MIN_HOURS = 4

# Impossible energy: more than this many times the home's contracted power, in one hour.
# Honest 2018 maximum: 1.29 times (12 homes went slightly over, mostly for a single hour).
IMPOSSIBLE_KWH_FACTOR = 1.5

# "Nearly all", the same in every street rule:
#   - outage: nearly all of a phase's homes show no voltage while the transformer phase reads 0;
#   - transformer meter wrong: nearly all of them show voltage while the transformer phase reads 0;
#   - all meters missing: nearly all of the street's meters sent nothing (one data link failure, not many
#     separate broken meters).
# One or two odd meters (e.g. stuck on their last value) can't block a clear verdict. When the homes disagree
# among themselves (between 10% and 90%, e.g, 45%), no street rule fires: the evidence contradicts itself.
NEARLY_ALL_SHARE = 0.9

# ----------------------------------------------------------------------------- input checks (plan step 1)
# What the median reading must look like, to catch wrong units.
VOLTAGE_MEDIAN_RANGE = (200.0, 260.0)  # volts on a 230 V network (kilovolts would be near 0.23)
KWH_MEDIAN_RANGE = (0.01, 5.0)  # kWh per hour for a home (Wh would be in the hundreds)

# ----------------------------------------------------------------------------- verdict names
# Given by the definition rules. Same names as the dataset's expected verdicts, so matching is exact.
VERDICT_MISSING_ONE = "missing_data_one_meter"
VERDICT_MISSING_ALL = "missing_data_all_meters"
VERDICT_IMPOSSIBLE = "impossible_value"
VERDICT_STUCK = "stuck_meter"
VERDICT_OUTAGE_PHASE = "outage_phase"
VERDICT_OUTAGE_STREET = "outage_street"
VERDICT_REFERENCE_METER = "reference_meter_problem"  # the transformer's own meter is wrong

# ----------------------------------------------------------------------------- scopes
# Where a finding sits.
# home, phase and street: same names as the dataset's scope column.
# transformer: the transformer's own meter (a problem the dataset never injects).
SCOPE_HOME = "home"
SCOPE_PHASE = "phase"
SCOPE_STREET = "street"
SCOPE_TRANSFORMER = "transformer"

# ----------------------------------------------------------------------------- trust states (plan step 4)
# Every home row and phase row in the memory carries one of these. Only trusted rows feed "usual".
#   trusted:    checked and found normal, a real change in use, or an estimated reading (the utility's own value)
#   untrusted:  a rule caught it, a street finding covers it, or it was diagnosed as a fault or "unknown"
#   unchecked:  one of its columns couldn't be computed, so nobody knows whether it's normal
TRUST_TRUSTED = "trusted"
TRUST_UNTRUSTED = "untrusted"
TRUST_UNCHECKED = "unchecked"
