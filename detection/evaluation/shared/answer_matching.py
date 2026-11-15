"""Matches the detector's incidents with the answer key."""

# To be built: an incident matches a problem if it covers the same home(s) or phase and overlaps in time;
# the verdict is right if it equals the expected one; for holidays and new ACs, success means no alert;
# an incident that matches no problem is a false alarm.
