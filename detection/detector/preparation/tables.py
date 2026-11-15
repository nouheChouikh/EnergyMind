"""Joins readings with wiring and day type, and builds the phase table."""

# To be built: one row per home and hour (with its phase, cable and type of day: weekday,
# or weekend-or-holiday), and one row per phase and hour (transformer energy on the phase,
# sum of its meters' kWh, how many meters reported).
