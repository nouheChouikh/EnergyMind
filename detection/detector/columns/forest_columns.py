"""The five columns for one hour: what the forest looks at for every home.

Why a model and not if-else rules: each column alone is ambiguous. A home using less can be a family
on holiday or a theft; a home's voltage dropping can be the grid or a broken meter. Only the columns
together tell them apart (less consumption + energy missing on the phase = suspicious; less consumption
+ energy adding up = normal life). The forest learns from a year of normal hours which combinations are
normal, including combinations nobody wrote a rule for. An if-else rule for every combination of five
columns can't be written; the forest scores the whole row at once.
"""

# To be built, one value per home and hour:
#   consumption_vs_usual                  is this home using more or less than usual? (holiday, new AC, theft)
#   voltage_gap_with_neighbours_vs_usual  is this meter's voltage out of line with its neighbours? (sensor drift)
#   phase_unexplained_energy              does the phase's energy add up, after cable losses? (theft vs holiday)
#   transformer_voltage_vs_usual          is the transformer's voltage off? (sag or surge: grid, not house)
#   hour                                  what time is it? (night and day differ)
