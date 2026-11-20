# Preprocessing Pipeline

This directory contains scripts for converting raw wearable device data into analysis-ready features.

## Pipeline Overview

```
Raw Device Data → Signal Processing → Assessment Alignment → Feature Extraction → Analysis Dataset
```

## Scripts

### 1. `process_e4.py` - Empatica E4 Signal Processing

Processes raw Empatica E4 data (sessions 1-15):
- **ACC.csv:** 3-axis accelerometer (32 Hz) → motion magnitude
- **BVP.csv:** Blood volume pulse (64 Hz) → BVP amplitude envelope (Hilbert transform)
- **EDA.csv:** Electrodermal activity (4 Hz) → phasic decomposition (cvxEDA), SCR detection
- **TEMP.csv:** Skin temperature (4 Hz) → median filtering
- **IBI.csv:** Inter-beat intervals → heart rate (cleaned, physiological range 300-2000 ms)

**Robust scaling:** Within-session median/IQR, capped at ±5 to prevent outlier-driven extrapolation.

### 2. `process_embraceplus.py` - EmbracePlus Signal Processing

Processes raw EmbracePlus Avro data (sessions 16-24):
- Parses Avro binary format (accelerometer, BVP, EDA, temperature, tags)
- Same signal processing as E4
- Handles different sampling rates (E4: EDA 4 Hz, EmbracePlus: EDA 1 Hz)

**Note:** EmbracePlus EDA was near sensor floor (median ≤ 0.01 µS) in sessions 16, 17, 18, 19, 21, 23 but retained for completeness.

### 3. `align_ratings.py` - Pain Assessment Alignment

Aligns pain scores with physiological signals:
- Reads manual rating sheets (CSV with timestamp, patient/nurse/physician scores)
- Corrects for device clock drift (if wristband time differs from wall-clock time)
- Flags assessments that fall outside the recording window
- Assigns assessment category (baseline / procedural / post)

**Output:** `aligned_assessments.csv` with columns:
```
subject, session, event_idx, timestamp, category, min_from_proc_start,
pain_patient, pain_nurse, pain_physician, in_recording
```

### 4. `extract_features.py` - Time-Domain Feature Extraction

For each assessment, extracts 37 features:

**Windows:**
- **Response:** −30 to +90 sec around recorded time (120 sec total)
- **Reference:** −180 to −30 sec (150 sec baseline)

**Feature definitions (6 stats × 5 modalities + 2 EDA-specific):**
- `level`: mean of response window
- `max`: maximum value in response window
- `std`: standard deviation
- `slope`: linear trend (OLS regression)
- `delta`: response mean minus reference mean
- `peak_rise`: response max minus reference mean

**EDA-specific:**
- `eda_scr_n`: skin conductance response count (response window)
- `eda_scr_amp`: mean SCR amplitude

**Context features (5):**
- `ctx_baseline`, `ctx_procedural`, `ctx_post`: one-hot encoding of assessment category
- `ctx_event_idx`: sequential assessment number (0-indexed)
- `ctx_min_from_proc_start`: minutes elapsed since procedure start

**Total:** 32 physiological + 5 context = 37 features

### 5. `build_dataset.py` - Analysis Dataset Assembly

Combines features from all sessions into `events_features.csv`:

```csv
subject,session,patient,event_idx,category,min_from_proc_start,device,
pain_patient,pain_nurse,pain_physician,
eda_level,eda_max,eda_std,eda_slope,eda_delta,eda_peak_rise,eda_scr_n,eda_scr_amp,
hr_level,hr_max,hr_std,hr_slope,hr_delta,hr_peak_rise,
temp_level,temp_max,temp_std,temp_slope,temp_delta,temp_peak_rise,
bvp_amp_level,bvp_amp_max,bvp_amp_std,bvp_amp_slope,bvp_amp_delta,bvp_amp_peak_rise,
motion_level,
ctx_baseline,ctx_procedural,ctx_post,ctx_event_idx,ctx_min_from_proc_start,
in_recording
```

**Patient ID mapping:** Maps session numbers to unique patient IDs (some patients contributed 2-4 sessions).

**Session inventory:**
- 24 sessions recorded
- 21 with analyzable wristband data (186 of 187 assessments)
- 3 excluded: session 10 (duplicate of 9), session 11 (no data), session 15 (download failed)

## Signal Processing Details

### EDA: Phasic Decomposition (cvxEDA)

Decomposes raw EDA into tonic (slow baseline drift) and phasic (event-related responses) components:

```python
import cvxopt
# Optimization problem: minimize ||y - (tonic + phasic)||² + λ ||∇²tonic||² + γ ||phasic||₁
# where y = raw EDA, tonic = smooth baseline, phasic = sparse transient responses
```

**SCR Detection:**
1. Apply cvxEDA to get phasic component
2. Find peaks with amplitude > 0.01 µS
3. Require minimum 1-sec separation between peaks

### Heart Rate: IBI Cleaning

1. Filter IBIs to physiological range (300-2000 ms, i.e., HR 30-200 bpm)
2. Compute instantaneous HR = 60000 / IBI (ms)
3. Interpolate to 1 Hz for feature extraction
4. Flag windows with <50% valid beats

**RMSSD (removed in revision):** Root mean square of successive differences (time-domain HRV). Was computed as `sqrt(mean(diff(IBI)²))` but removed after reviewer flagged short-window unreliability.

### BVP Amplitude: Envelope Detection

1. Apply Hilbert transform to BVP signal
2. Compute instantaneous amplitude as `abs(analytic_signal)`
3. Low-pass filter envelope (1 Hz cutoff) to remove beat-to-beat fluctuations
4. Downsample to 1 Hz

### Temperature: Median Filtering

1. Apply 5-sample median filter to remove sensor noise
2. Downsample to 1 Hz (E4: from 4 Hz; EmbracePlus: already at 1 Hz)

### Motion: Accelerometer Magnitude

1. Compute 3-axis magnitude: `sqrt(x² + y² + z²)`
2. High-pass filter (0.5 Hz) to remove gravity
3. Low-pass filter (5 Hz) to remove sensor noise
4. Downsample to 1 Hz

## Missing Values

Features may be missing if:
- Assessment falls partially outside recording window
- Signal quality is poor (e.g., no skin contact for EDA)
- Beats are too sparse for HRV (flagged if <50% valid)

**Handling:** Imputed with training-fold median during model fitting (never using held-out data).

## Data Quality Notes

From `session_inventory.csv`:
- **E4 beat stream too sparse:** session 4 (0% RMSSD coverage), session 6 (20%)
- **EDA flat at 0:** session 13 (no skin contact)
- **EmbracePlus EDA near floor:** sessions 16, 17, 18, 19, 21, 23 (median ≤ 0.01 µS)
- **Rating sheet data integrity:** session 23 (physician = patient at all 16 events), session 17 (nurse = patient 11/12 events) — verified against paper records

## Running the Pipeline

**Note:** These scripts require raw data that are not included in this repository. They are provided for transparency and reproducibility.

```bash
# Step 1: Process device signals
python preprocessing/process_e4.py --input ../data/raw --output ../data/processed
python preprocessing/process_embraceplus.py --input ../data/raw --output ../data/processed

# Step 2: Align pain assessments
python preprocessing/align_ratings.py \
    --signals ../data/processed \
    --ratings ../data/raw \
    --output ../data/processed

# Step 3: Extract features
python preprocessing/extract_features.py \
    --input ../data/processed \
    --output ../data/processed

# Step 4: Build analysis dataset
python preprocessing/build_dataset.py \
    --input ../data/processed \
    --output ../data/processed/events_features.csv
```

Expected runtime: ~10 minutes total for 21 sessions.

## Output Files

- `processed/subject_NN_signals.csv`: 1 Hz time series (eda, hr, temp, bvp_amp, motion)
- `processed/subject_NN_scrs.csv`: Detected skin conductance responses
- `processed/aligned_assessments.csv`: Pain scores aligned to signal timestamps
- `processed/events_features.csv`: **Final analysis dataset** (37 features × N assessments)
- `processed/session_inventory.csv`: Per-session metadata (device, n_events, quality flags)
