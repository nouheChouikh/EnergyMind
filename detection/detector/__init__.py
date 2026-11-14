"""The EnergyMind detection system: learns what a normal street looks like, then checks one hour at a time.

It never runs on its own: callers (evaluation now, the operator app's backend later) pass tables in
and get results back.
"""

# Front door: the public functions (learn, start_memory, check_hour, apply_operator_decision,
# save_model, load_model) are imported here as each part is built.
