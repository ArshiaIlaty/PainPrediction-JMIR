"""
Re-analysis of the BSN 2026 submission "Beyond the Sweat Response" (HRV suppression claim),
addressing the reviewers' requests on the full cohort.

Per event: ultra-short PPG-derived pulse-rate variability (RMSSD, pNN50; "PRV") and phasic EDA
(SCR count / max amplitude / area) in pre-specified windows:
  paper window  [-15 s, +15 s]   (as in the submission)
  robust window [-30 s, +90 s]   (covers minute-resolution clinical timestamps)
  local reference [-180 s, -30 s] -> change scores (delta ln RMSSD)
Beats: E4 IBI.csv (device beat detector, subjects 1-14); EmbracePlus systolic peaks (16-24).
IBI cleaning: 300-2000 ms and <20 % change from the running median of 5.

Statistics (events nested in sessions nested in patients):
  - pooled Spearman with patient-cluster bootstrap CI (what the paper reported, done properly)
  - repeated-measures correlation (within-session centring; df = N - sessions - 1)
  - within-session permutation test (shuffle scores within session)
  - linear mixed model: score ~ z(lnRMSSD) [+ event type + time] + (1 | patient), session VC
  - session-level response score S_HRV with bootstrap CI over events
Label definitions: patient (primary), nurse, physician, mean, max (submission's definition).
Subsets: all scored non-baseline events (submission), procedural only (needles/lidocaine);
cohorts: EmbracePlus 16-24 (submission), E4 1-14, all.

Usage: python event_pipeline/hrv_analysis.py
Writes out/hrv_events.csv, out/hrv_results.csv, out/HRV_REPORT.md
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.signal import find_peaks
from scipy.stats import pearsonr, spearmanr, t as tdist

import build_dataset as B
from run_experiments import PATIENT

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent / "out"
WINDOWS = {"paper_pm15": (-15, 15), "robust_m30p90": (-30, 90)}
REF = (-180, -30)
RATERS = ["pain_patient", "pain_nurse", "pain_physician", "pain_mean", "pain_max"]
RNG = np.random.default_rng(0)


# ----------------------------------------------------------------------------- beats
def beats_e4(sid):
    out = []
    for f in B.subject_dir(sid).rglob("IBI.csv"):
        if f.stat().st_size == 0:      # E4 writes empty IBI files when no beat passes its QC
            continue
        raw = pd.read_csv(f, header=None)
        if len(raw) < 3:
            continue
        t0 = float(raw.iloc[0, 0])
        v = raw.iloc[1:].to_numpy(float)
        out.append(np.c_[t0 + v[:, 0], v[:, 1]])
    a = np.vstack(out) if out else np.empty((0, 2))
    return a[np.argsort(a[:, 0])]


def beats_embrace(sid):
    pk = pd.read_csv(B.EMB / f"subject_{sid}" / "raw_systolic_peaks.csv",
                     usecols=["systolic_peak_timestamp"])
    tp = np.sort(pk.systolic_peak_timestamp.to_numpy(np.int64)) / 1e9
    return np.c_[tp[1:], np.diff(tp)]


def clean_ibi(b):
    """b: (n, 2) [beat time s, ibi s]. Physiological range + successive-change filter.
    Returns cleaned array plus a flag marking intervals that follow a gap (no successive diff)."""
    ok = (b[:, 1] > 0.3) & (b[:, 1] < 2.0)
    b = b[ok]
    if len(b) < 5:
        return b, np.zeros(len(b), bool)
    med = pd.Series(b[:, 1]).rolling(5, center=True, min_periods=1).median().to_numpy()
    b = b[np.abs(b[:, 1] - med) / med < 0.2]
    # E4 drops beats it cannot detect: only take successive differences across true neighbours
    contiguous = np.r_[False, np.abs(np.diff(b[:, 0]) - b[1:, 1]) < 0.1]
    return b, contiguous


def prv(b, contiguous, lo, hi, min_pairs=8):
    m = (b[:, 0] >= lo) & (b[:, 0] <= hi)
    idx = np.where(m)[0]
    pairs = [i for i in idx if contiguous[i] and i - 1 >= 0 and m[i - 1]]
    n_beats = int(m.sum())
    if len(pairs) < min_pairs:
        return {"rmssd": np.nan, "pnn50": np.nan, "n_beats": n_beats, "n_pairs": len(pairs)}
    d = (b[pairs, 1] - b[[i - 1 for i in pairs], 1]) * 1000
    return {"rmssd": float(np.sqrt(np.mean(d ** 2))), "pnn50": float(np.mean(np.abs(d) > 50) * 100),
            "n_beats": n_beats, "n_pairs": len(pairs)}


def scr_features(eda, t, lo, hi, fs=4.0):
    """Phasic EDA on the 4 Hz signal: high-pass via 4 s median detrend, peaks prominence 0.01 uS."""
    w = eda.loc[(eda.index >= t + lo) & (eda.index <= t + hi)]
    if len(w) < (hi - lo) * fs * 0.5:
        return {"scr_n": np.nan, "scr_amp": np.nan, "scr_area": np.nan}
    x = w.to_numpy()
    phasic = x - pd.Series(x).rolling(int(4 * fs), center=True, min_periods=1).median().to_numpy()
    pk, prop = find_peaks(x, prominence=0.01, distance=int(1 * fs))
    return {"scr_n": len(pk), "scr_amp": float(prop["prominences"].max()) if len(pk) else 0.0,
            "scr_area": float(np.clip(phasic, 0, None).sum() / fs)}


def eda_4hz(sid):
    if sid in B.E4_SUBJECTS:
        parts = []
        for f in B.subject_dir(sid).rglob("EDA.csv"):
            t, v = B.read_e4(f)
            parts.append(pd.Series(v[:, 0], index=t))
        return pd.concat(parts).sort_index()
    e = pd.read_csv(B.EMB / f"subject_{sid}" / "raw_eda.csv", usecols=["timestamp_unix_ms", "eda"])
    return pd.Series(e.eda.to_numpy(), index=e.timestamp_unix_ms.to_numpy() / 1e3).sort_index()


# ----------------------------------------------------------------------------- build
def build():
    rows = []
    for sid in B.E4_SUBJECTS + B.EMB_SUBJECTS:
        b = beats_e4(sid) if sid in B.E4_SUBJECTS else beats_embrace(sid)
        b, contig = clean_ibi(b)
        eda = eda_4hz(sid)
        ev = B.load_events(sid, eda.index.min())
        proc_t = ev.loc[ev.category == "procedural", "t_utc"]
        ev["min_from_proc_start"] = (ev.t_utc - (proc_t.min() if len(proc_t) else ev.t_utc.min())) / 60
        ev = ev[(ev.t_utc >= eda.index.min()) & (ev.t_utc <= eda.index.max())]
        for _, e in ev.iterrows():
            r = {**e.to_dict(), "clock_has_seconds": int(round(e.t_utc)) % 60 != 0}
            ref = prv(b, contig, e.t_utc + REF[0], e.t_utc + REF[1], min_pairs=20)
            r["ref_rmssd"] = ref["rmssd"]
            for wname, (lo, hi) in WINDOWS.items():
                p = prv(b, contig, e.t_utc + lo, e.t_utc + hi)
                for k, v in p.items():
                    r[f"{wname}_{k}"] = v
                for k, v in scr_features(eda, e.t_utc, lo, hi).items():
                    r[f"{wname}_{k}"] = v
                r[f"{wname}_dlnrmssd"] = np.log(p["rmssd"]) - np.log(ref["rmssd"]) \
                    if p["rmssd"] > 0 and ref["rmssd"] > 0 else np.nan
            rows.append(r)
        print(f"subject {sid:2d}: {len(ev)} events, clean beats {len(b)}, "
              f"RMSSD available (paper window) "
              f"{np.mean([not np.isnan(x['paper_pm15_rmssd']) for x in rows[-len(ev):]]) if len(ev) else 0:.0%}",
              flush=True)
    d = pd.DataFrame(rows)
    d["patient"] = d.subject.map(PATIENT)
    d["pain_mean"] = d[["pain_patient", "pain_nurse", "pain_physician"]].mean(axis=1)
    d["pain_max"] = d[["pain_patient", "pain_nurse", "pain_physician"]].max(axis=1)
    for w in WINDOWS:
        d[f"{w}_lnrmssd"] = np.log(d[f"{w}_rmssd"])
    return d


# ----------------------------------------------------------------------------- stats
def _rank(v):
    return pd.Series(v).rank().to_numpy()


def _spearman(x, y):
    if len(x) < 5 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return np.corrcoef(_rank(x), _rank(y))[0, 1]


def _rmcorr(x, y, sess):
    """Repeated-measures correlation via within-session centring (numpy, codes 0..S-1)."""
    S = sess.max() + 1
    cnt = np.bincount(sess, minlength=S)
    xc = x - (np.bincount(sess, x, S) / np.maximum(cnt, 1))[sess]
    yc = y - (np.bincount(sess, y, S) / np.maximum(cnt, 1))[sess]
    if np.std(xc) == 0 or np.std(yc) == 0:
        return np.nan
    return np.corrcoef(xc, yc)[0, 1]


def pooled_spearman(d, x, y):
    m = d[[x, y]].dropna()
    return _spearman(m[x].to_numpy(float), m[y].to_numpy(float))


def prep(d, x, y):
    m = d[["patient", "subject", x, y]].dropna()
    m = m[m.groupby("subject")[y].transform("std") > 0]
    return (m[x].to_numpy(float), m[y].to_numpy(float), pd.factorize(m.subject)[0],
            pd.factorize(m.patient)[0], m)


def cluster_boot(xv, yv, sess, pat, n=1000):
    """Patient-cluster bootstrap of pooled Spearman and rmcorr (resampled sessions relabelled)."""
    rows = [np.where(pat == p)[0] for p in range(pat.max() + 1)]
    sp, rm = [], []
    for _ in range(n):
        pick = RNG.integers(0, len(rows), len(rows))
        idx = np.concatenate([rows[p] for p in pick])
        # each resampled copy of a patient gets its own session codes
        lab = np.concatenate([sess[rows[p]] + k * (sess.max() + 1) for k, p in enumerate(pick)])
        sp.append(_spearman(xv[idx], yv[idx]))
        rm.append(_rmcorr(xv[idx], yv[idx], pd.factorize(lab)[0]))
    return np.nanpercentile(sp, [2.5, 97.5]), np.nanpercentile(rm, [2.5, 97.5])


def rmcorr(d, x, y):
    xv, yv, sess, _, m = prep(d, x, y)
    if m.subject.nunique() < 2:
        return np.nan, np.nan, 0
    r = _rmcorr(xv, yv, sess)
    df = len(m) - m.subject.nunique() - 1
    tval = r * np.sqrt(df / max(1e-12, 1 - r ** 2))
    return r, 2 * tdist.sf(abs(tval), df), df


def within_perm_p(d, x, y, n=2000):
    xv, yv, sess, _, _ = prep(d, x, y)
    obs = _rmcorr(xv, yv, sess)
    idx = [np.where(sess == s)[0] for s in range(sess.max() + 1)]
    yp = yv.copy()
    null = []
    for _ in range(n):
        for i in idx:
            yp[i] = RNG.permutation(yp[i])
        null.append(_rmcorr(xv, yp, sess))
    return float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n + 1))


def mixed(d, x, y, adjust):
    m = d[["patient", "subject", "category", "min_from_proc_start", x, y]].dropna().copy()
    if m.patient.nunique() < 3:
        return np.nan, np.nan, np.nan, np.nan
    m["xz"] = (m[x] - m[x].mean()) / m[x].std()
    f = f"{y} ~ xz" + (" + C(category) + min_from_proc_start" if adjust else "")
    try:
        r = smf.mixedlm(f, m, groups="patient", vc_formula={"session": "0 + C(subject)"}).fit(
            reml=True, method="lbfgs")
        ci = r.conf_int().loc["xz"]
        return r.params["xz"], ci[0], ci[1], r.pvalues["xz"]
    except Exception:
        return np.nan, np.nan, np.nan, np.nan


def session_scores(d, x, y, thr=4):
    """S = mean z(x | high) - mean z(x | low), z within session; bootstrap CI over events."""
    rows = []
    for s, g in d.groupby("subject"):
        g = g[[x, y]].dropna()
        hi, lo = g[g[y] >= thr], g[g[y] < thr]
        if len(hi) < 2 or len(lo) < 2 or g[x].std() == 0:
            rows.append({"subject": s, "n_high": len(hi), "n_low": len(lo), "S": np.nan,
                         "ci_low": np.nan, "ci_high": np.nan})
            continue
        z = (g[x] - g[x].mean()) / g[x].std()
        zh, zl = z[g[y] >= thr].to_numpy(), z[g[y] < thr].to_numpy()
        bs = [RNG.choice(zh, len(zh)).mean() - RNG.choice(zl, len(zl)).mean() for _ in range(2000)]
        rows.append({"subject": s, "n_high": len(hi), "n_low": len(lo), "S": zh.mean() - zl.mean(),
                     "ci_low": np.percentile(bs, 2.5), "ci_high": np.percentile(bs, 97.5)})
    return pd.DataFrame(rows)


def analyse(d):
    cohorts = {"EmbracePlus_16-24": d[d.device == "EmbracePlus"], "E4_1-14": d[d.device == "E4"],
               "all": d}
    subsets = {"scored_non_baseline": lambda x: x[x.category != "baseline"],
               "procedural_only": lambda x: x[x.category == "procedural"]}
    feats = ["paper_pm15_lnrmssd", "paper_pm15_pnn50", "robust_m30p90_lnrmssd",
             "robust_m30p90_dlnrmssd", "paper_pm15_scr_amp", "robust_m30p90_scr_amp",
             "robust_m30p90_scr_area"]
    rows = []
    for cname, c in cohorts.items():
        for sname, sf in subsets.items():
            dd = sf(c)
            for f in feats:
                for y in RATERS:
                    m = dd[["patient", "subject", "category", "min_from_proc_start", f, y]].dropna()
                    if len(m) < 15:
                        continue
                    rho = pooled_spearman(m, f, y)
                    r_rm, p_rm, df = rmcorr(m, f, y)
                    (lo, hi), (rlo, rhi) = cluster_boot(*prep(m, f, y)[:4])
                    b, blo, bhi, bp = mixed(m, f, y, adjust=False)
                    ba, balo, bahi, bap = mixed(m, f, y, adjust=True)
                    rows.append({"cohort": cname, "subset": sname, "feature": f, "rater": y,
                                 "n_events": len(m), "n_sessions": m.subject.nunique(),
                                 "n_patients": m.patient.nunique(),
                                 "pooled_rho": rho, "pooled_rho_ci": f"[{lo:.2f}, {hi:.2f}]",
                                 "rm_r": r_rm, "rm_r_ci": f"[{rlo:.2f}, {rhi:.2f}]", "rm_p": p_rm,
                                 "lmm_beta": b, "lmm_ci": f"[{blo:.2f}, {bhi:.2f}]", "lmm_p": bp,
                                 "lmm_adj_beta": ba, "lmm_adj_ci": f"[{balo:.2f}, {bahi:.2f}]",
                                 "lmm_adj_p": bap})
            print(f"  analysed {cname} / {sname}", flush=True)
    return pd.DataFrame(rows)


def main():
    d = build()
    d.to_csv(OUT / "hrv_events.csv", index=False)
    res = analyse(d)
    # within-session permutation p for the headline rows (paper window RMSSD)
    key = res.feature.isin(["paper_pm15_lnrmssd", "robust_m30p90_lnrmssd", "paper_pm15_scr_amp"])
    for i in res[key].index:
        r = res.loc[i]
        c = {"EmbracePlus_16-24": d[d.device == "EmbracePlus"], "E4_1-14": d[d.device == "E4"],
             "all": d}[r.cohort]
        c = c[c.category != "baseline"] if r.subset == "scored_non_baseline" else c[c.category == "procedural"]
        res.loc[i, "within_perm_p"] = within_perm_p(c, r.feature, r.rater, n=1000)
    res.to_csv(OUT / "hrv_results.csv", index=False)
    sess = []
    for y in ["pain_max", "pain_patient"]:
        for w in WINDOWS:
            s = session_scores(d[(d.device == "EmbracePlus") & (d.category != "baseline")],
                               f"{w}_lnrmssd", y)
            s["label"], s["window"] = y, w
            sess.append(s)
    sess = pd.concat(sess)
    sess.to_csv(OUT / "hrv_session_scores.csv", index=False)
    write_report(d, res, sess)


def write_report(d, res, sess):
    L = ["# HRV-suppression re-analysis (BSN 2026 submission)", ""]
    cov = d.groupby("device").agg(events=("subject", "size"), sessions=("subject", "nunique"),
                                  clock_with_seconds=("clock_has_seconds", "mean"),
                                  rmssd_avail_pm15=("paper_pm15_rmssd", lambda v: v.notna().mean()),
                                  rmssd_avail_robust=("robust_m30p90_rmssd", lambda v: v.notna().mean()),
                                  median_beats_pm15=("paper_pm15_n_beats", "median"))
    L += ["## Data coverage", "", cov.round(2).to_markdown(), "",
          "## Associations (rm_r = within-session repeated-measures correlation; LMM beta = NRS "
          "points per 1 SD of feature; adj = + event type + time)", ""]
    show = res[res.feature.isin(["paper_pm15_lnrmssd", "robust_m30p90_lnrmssd",
                                 "robust_m30p90_dlnrmssd", "paper_pm15_scr_amp",
                                 "robust_m30p90_scr_amp", "robust_m30p90_scr_area"])]
    cols = ["cohort", "subset", "feature", "rater", "n_events", "n_patients", "pooled_rho",
            "pooled_rho_ci", "rm_r", "rm_r_ci", "rm_p", "within_perm_p", "lmm_beta", "lmm_ci",
            "lmm_p", "lmm_adj_beta", "lmm_adj_ci", "lmm_adj_p"]
    L += [show[cols].round(3).to_markdown(index=False), "",
          "## Session-level S_HRV (EmbracePlus, scored non-baseline events)", "",
          sess.round(2).to_markdown(index=False), ""]
    (OUT / "HRV_REPORT.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
