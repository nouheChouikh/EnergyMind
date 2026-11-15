"""What is carried from one hour to the next."""

# To be built: the recent readings (last 28 days + 24 hours), which hours are trusted, and the open incidents.
# Also, for each home and for the transformer: the last usual that could be computed, per hour of day
# and type of day; used when fewer than 3 trusted values remain (e.g. during a long theft).
# All plain tables, so the caller can keep them: in RAM for evaluation, in a database for the app.
