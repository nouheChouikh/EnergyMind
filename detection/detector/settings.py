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

# ----------------------------------------------------------------------------- definition rules (plan step 2)
# Stuck meter: the whole reading (kWh, V and I) identical this many hours in a row.
# Honest meters repeat their whole reading at most 3 hours in a row (dataset checks, 2018).
STUCK_MIN_HOURS = 4

# Impossible energy: more than this many times the home's contracted power, in one hour.
# Honest 2018 maximum: 1.29 times (12 homes went slightly over, mostly for a single hour).
IMPOSSIBLE_KWH_FACTOR = 1.5

# All meters missing: at least this share of the street's meters empty in the same hour.
# Nearly all at once means one collection failure, not many separate broken meters.
MISSING_ALL_SHARE = 0.9

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
