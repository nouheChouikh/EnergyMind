"""Which home rows and phase rows count as trusted (plan step 4)."""

# To be built: trust on two levels. Home rows feed the home's usuals; phase rows feed the grid voltage usual.
# A row is untrusted if a rule caught it, or if it was diagnosed as a fault or "unknown". A grid event
# untrusts everything it touches (its phase row and every home row on that phase); a home problem untrusts
# only that home's row. Hours diagnosed as a real change in use stay trusted. A row whose columns can't all
# be computed is unchecked. Operator decisions can change trust later (feedback/operator_decisions.py).
