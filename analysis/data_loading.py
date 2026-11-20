"""
Data loading and feature set definitions for JMIR analysis.

Expects: ../data/processed/events_features.csv
"""

from pathlib import Path
import pandas as pd

# Patient ID mapping (session -> unique patient ID)
PATIENT_MAP = {
    1: "P01", 3: "P01", 8: "P01",  # Patient 1: 3 sessions
    2: "P02", 5: "P02",            # Patient 2: 2 sessions
    4: "P03",                      # Patient 3: 1 session
    6: "P04",                      # Patient 4: 1 session
    7: "P05",                      # Patient 5: 1 session
    9: "P06",                      # Patient 6: 1 session
    12: "P07", 13: "P07",          # Patient 7: 2 sessions
    14: "P08",                     # Patient 8: 1 session
    16: "P09", 17: "P09", 18: "P09", 20: "P09",  # Patient 9: 4 sessions
    19: "P10", 21: "P10",          # Patient 10: 2 sessions
    22: "P11",                     # Patient 11: 1 session
    23: "P12",                     # Patient 12: 1 session
    24: "P13"                      # Patient 13: 1 session
}


def load():
    """
    Load events_features.csv and add patient IDs.

    Returns:
        pd.DataFrame with columns:
            - subject (session number)
            - patient (unique patient ID P01-P13)
            - event_idx, category, min_from_proc_start, device
            - pain_patient, pain_nurse, pain_physician
            - 32 physiological features (eda, hr, temp, bvp_amp, motion)
            - 5 context features (ctx_baseline, ctx_procedural, ctx_post, ctx_event_idx, ctx_min_from_proc_start)
            - quality flags (in_recording, etc.)
    """
    data_path = Path(__file__).parents[1] / "data" / "processed" / "events_features.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} not found. Please run preprocessing pipeline first, or "
            f"contact the authors for data access (see README.md)."
        )

    d = pd.read_csv(data_path)

    # Add patient IDs
    d["patient"] = d["subject"].map(PATIENT_MAP)

    # Robust scaling is already applied in preprocessing
    # (within-session median/IQR, capped at ±5)

    # Filter to assessments that fall inside recording window
    d = d[d.in_recording].reset_index(drop=True)

    return d


def feature_sets(d):
    """
    Define feature sets for analysis.

    Returns:
        dict with keys:
            - "physio_no_motion": physiological features WITHOUT motion (32 features)
            - "motion": motion only (1 feature)
            - "context": procedural context (5 features)
            - "physio_no_motion+context": combined (37 features)
    """
    # Physiological features (excluding motion)
    physio_no_motion = [c for c in d.columns if any(c.startswith(p) for p in ["eda_", "hr_", "temp_", "bvp_amp_"])]

    # Motion (reported separately as confound)
    motion = ["motion_level"]

    # Context features
    context = [c for c in d.columns if c.startswith("ctx_")]

    return {
        "physio_no_motion": physio_no_motion,
        "motion": motion,
        "context": context,
        "physio_no_motion+context": physio_no_motion + context
    }
