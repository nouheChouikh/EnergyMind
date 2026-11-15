"""Replays 2019 through the detector and measures it on the practice months (the odd months).

Run from the repository root, with .venv active, as often as needed:
    python detection/evaluation/run_practice.py

Used to choose the few settings (T, the naming rule, incident closing) and to fill the confidence table.
Writes detection/outputs/practice_results/, and the confidence table into detection/outputs/saved_model/.
"""

# To be built:
# 1. load the trained model from detection/outputs/saved_model/
# 2. start_memory() from the last 28 days of 2018
# 3. replay every hour of 2019 in order through check_hour() (shared/replay.py)
# 4. keep only the practice months and match their incidents with the answer key (shared/answer_matching.py)
# 5. compute the measures (shared/measures.py), run the threshold baseline the same way
#    (shared/threshold_baseline.py), and write the report (shared/report_writing.py)
# 6. write the confidence table (how often each verdict type was right) into detection/outputs/saved_model/
