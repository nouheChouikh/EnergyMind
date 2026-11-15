"""Replays 2019 through the detector and measures it on the final-test months (the even months).

Run from the repository root, with .venv active, once, at the very end:
    python detection/evaluation/run_final_test.py

Nothing is tuned or changed here, not even the confidence. Refuses to run if
detection/outputs/final_test_results/ already exists. Writes detection/outputs/final_test_results/.
"""

# To be built:
# 1. stop if detection/outputs/final_test_results/ already exists (the final test runs once)
# 2. load the trained model from detection/outputs/saved_model/, with the confidence table from practice
# 3. start_memory() from the last 28 days of 2018
# 4. replay every hour of 2019 in order through check_hour() (shared/replay.py)
# 5. keep only the final-test months and match their incidents with the answer key (shared/answer_matching.py)
# 6. compute the measures (shared/measures.py), held-out problems included, run the threshold baseline
#    the same way (shared/threshold_baseline.py), and write the report (shared/report_writing.py)
# 7. check that each verdict type's confidence (from practice) holds up: compare it with how often
#    that verdict was right here
