"""Which home explains missing energy on a phase."""

# To be built: compares the energy missing over the last 24 hours with each home's drop below its usual,
# closest match first.
# Limitation: a home is named only if the match is decent: its drop is about the same as the missing
# energy (how close: a setting tuned in practice). A thief can hide a brand-new appliance behind a bypass:
# their recorded use doesn't drop, so the missing energy truly matches nobody, and naming the closest home
# would accuse an innocent one. If no home matches, nobody is named: energy_missing_unattributed, and a
# person decides. Even a decent match makes a home a suspect, never proof.
