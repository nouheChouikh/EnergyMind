"""Confidence per verdict type: how often this kind of verdict was right."""

# To be built: for each verdict type, (right + 1) / (total + 2), shown with the count
# (Laplace's rule of succession).
# The +1 and +2 keep small counts honest: 6 right out of 6 gives 88%, not 100%; no cases gives 50%.
# The effect fades as cases pile up (90 out of 90 gives 99%). "unknown" has no confidence.
