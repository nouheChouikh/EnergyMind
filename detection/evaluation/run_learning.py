"""Learns from the dataset's 2018 (the learning year) and saves the trained model.

Run from the repository root, with .venv active:
    python detection/evaluation/run_learning.py

Writes detection/outputs/saved_model/ (committed: the operator app loads the model from there).
"""

# To be built:
# 1. read 2018 from dataset/data/system_inputs/ (shared/dataset_files.py)
# 2. call the detector's learn(): losses regression, six columns, forest, T
# 3. save_model() into detection/outputs/saved_model/, with the settings and library versions used
