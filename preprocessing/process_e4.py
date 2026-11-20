"""
Empatica E4 Signal Processing

Converts raw E4 CSV files into 1 Hz processed time series with robust scaling.

Input directory structure:
    raw/subject_NN/E4_session/
        ACC.csv, BVP.csv, EDA.csv, TEMP.csv, IBI.csv, tags.csv, info.txt

Output:
    processed/subject_NN_signals.csv (columns: timestamp, eda, hr, temp, bvp_amp, motion)
    processed/subject_NN_scrs.csv (columns: timestamp, amplitude)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, hilbert, medfilt


def load_e4_csv(fpath, start_time=None):
    """Load E4 CSV with unix timestamp header."""
    with open(fpath) as f:
        ts = float(f.readline().strip())
        hz = float(f.readline().strip())
    data = pd.read_csv(fpath, skiprows=2, header=None, names=["value"])
    if start_time is None:
        start_time = ts
    data["timestamp"] = start_time + np.arange(len(data)) / hz
    return data


def butter_lowpass(sig, cutoff, fs, order=4):
    """Butterworth low-pass filter."""
    b, a = butter(order, cutoff / (0.5 * fs), btype="low")
    return filtfilt(b, a, sig)


def butter_highpass(sig, cutoff, fs, order=4):
    """Butterworth high-pass filter."""
    b, a = butter(order, cutoff / (0.5 * fs), btype="high")
    return filtfilt(b, a, sig)


def process_eda(eda_raw, hz=4):
    """EDA: low-pass filter, phasic decomposition (cvxEDA), SCR detection."""
    # Low-pass filter (1 Hz)
    eda_filt = butter_lowpass(eda_raw.value.to_numpy(), 1.0, hz)

    # cvxEDA phasic decomposition (requires cvxopt)
    # Placeholder: implement cvxEDA or use existing library
    phasic = np.zeros_like(eda_filt)  # TODO: actual cvxEDA

    # SCR detection: peaks in phasic > 0.01 µS, min 1 sec separation
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(phasic, height=0.01, distance=int(hz))
    scrs = pd.DataFrame({"timestamp": eda_raw.timestamp.iloc[peaks].values,
                         "amplitude": phasic[peaks]})

    return eda_filt, scrs


def process_bvp(bvp_raw, hz=64):
    """BVP: Hilbert envelope for amplitude, low-pass filter, downsample to 1 Hz."""
    bvp = bvp_raw.value.to_numpy()
    # Envelope via Hilbert transform
    analytic = hilbert(bvp)
    env = np.abs(analytic)
    # Low-pass filter envelope (1 Hz)
    env_filt = butter_lowpass(env, 1.0, hz)
    # Downsample to 1 Hz
    idx = np.arange(0, len(env_filt), hz).astype(int)
    return env_filt[idx], bvp_raw.timestamp.iloc[idx].values


def process_ibi(ibi_csv):
    """IBI: clean, convert to HR, interpolate to 1 Hz."""
    ibi = pd.read_csv(ibi_csv, names=["timestamp_offset", "ibi_ms"])
    # Filter to physiological range (300-2000 ms → HR 30-200 bpm)
    ibi = ibi[(ibi.ibi_ms >= 300) & (ibi.ibi_ms <= 2000)]
    # Instantaneous HR
    ibi["hr"] = 60000 / ibi.ibi_ms
    # Interpolate to 1 Hz (use absolute timestamps)
    # TODO: interpolate to reference time grid
    return ibi


def process_temp(temp_raw, hz=4):
    """Temperature: median filter, downsample to 1 Hz."""
    temp = medfilt(temp_raw.value.to_numpy(), kernel_size=5)
    idx = np.arange(0, len(temp), hz).astype(int)
    return temp[idx], temp_raw.timestamp.iloc[idx].values


def process_acc(acc_csv, hz=32):
    """Accelerometer: 3-axis magnitude, high-pass (0.5 Hz), low-pass (5 Hz), downsample to 1 Hz."""
    acc = pd.read_csv(acc_csv, skiprows=2, names=["x", "y", "z"])
    # Magnitude
    mag = np.sqrt((acc ** 2).sum(axis=1))
    # High-pass to remove gravity
    mag = butter_highpass(mag, 0.5, hz)
    # Low-pass to remove noise
    mag = butter_lowpass(mag, 5.0, hz)
    # Downsample
    idx = np.arange(0, len(mag), hz).astype(int)
    # Reconstruct timestamps (ACC has same structure as other E4 files)
    with open(acc_csv) as f:
        ts = float(f.readline().strip())
    timestamps = ts + idx / hz
    return mag[idx], timestamps


def robust_scale(df, signals):
    """Within-session robust scaling: (x - median) / IQR, capped at ±5."""
    for sig in signals:
        if sig in df.columns:
            med = df[sig].median()
            iqr = df[sig].quantile(0.75) - df[sig].quantile(0.25)
            if iqr > 0:
                df[sig] = ((df[sig] - med) / iqr).clip(-5, 5)
    return df


def process_subject(subject_dir, out_dir):
    """Process one E4 session."""
    e4_dir = subject_dir / "E4_session"
    if not e4_dir.exists():
        print(f"Skipping {subject_dir.name}: no E4 data")
        return

    # Load raw files
    eda_raw = load_e4_csv(e4_dir / "EDA.csv")
    bvp_raw = load_e4_csv(e4_dir / "BVP.csv")
    temp_raw = load_e4_csv(e4_dir / "TEMP.csv")

    # Process
    eda, scrs = process_eda(eda_raw)
    bvp_amp, bvp_ts = process_bvp(bvp_raw)
    temp, temp_ts = process_temp(temp_raw)
    motion, motion_ts = process_acc(e4_dir / "ACC.csv")
    hr_df = process_ibi(e4_dir / "IBI.csv")

    # Merge to common 1 Hz grid (TODO: implement proper interpolation)
    # df = merge_to_1hz_grid(...)

    # Robust scaling
    # df = robust_scale(df, ["eda", "hr", "temp", "bvp_amp", "motion"])

    # Save
    # df.to_csv(out_dir / f"{subject_dir.name}_signals.csv", index=False)
    # scrs.to_csv(out_dir / f"{subject_dir.name}_scrs.csv", index=False)

    print(f"Processed {subject_dir.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Directory with raw/subject_NN/")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for processed/")
    args = parser.parse_args()

    args.output.mkdir(exist_ok=True, parents=True)

    for subj_dir in sorted(args.input.glob("subject_*")):
        if subj_dir.is_dir():
            process_subject(subj_dir, args.output)


if __name__ == "__main__":
    main()
