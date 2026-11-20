"""
Per-session overview figures + session inventory for all 24 study sessions.

Figure per session with wristband data (figures/session_overviews/session_NN.png):
  top: pain ratings (patient / nurse / physician) at each assessment, numbered events, phases shaded
  then EDA, HR, PPG RMSSD (30-s), BVP amplitude, skin temperature, wrist motion on a shared local-time axis
  bottom: event-locked traces around every needle/lidocaine event (-60 to +120 s, minus the
          pre-event mean), colored by patient-reported pain; plus the event list
Inventory (out/session_inventory.csv): every session 1-24 with device, recording span, assessments in the
sheet / inside the recording, per-modality availability, and exclusion reasons.

Usage: python event_pipeline/session_overviews.py
"""

import warnings
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize

import build_dataset as B
import hrv_analysis as H
from run_experiments import PATIENT

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "out"
FIG = ROOT / "figures" / "session_overviews"
TZ = "America/Los_Angeles"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, WASH = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#f0efec"
RATERS = [("pain_patient", BLUE, "o", "Patient"), ("pain_nurse", ORANGE, "s", "Nurse"),
          ("pain_physician", AQUA, "^", "Physician")]
# sequential blue ramp (validated reference palette, ordinal steps 250-700) for pain 0-10
PAIN_CMAP = LinearSegmentedColormap.from_list("pain", ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
EXCLUDED = {10: "wristband folder is a byte-identical copy of session 9 (recorded 2024-02-23)",
            11: "no wristband data in the session folder (rating sheet and PDF only)",
            15: "wristband data could not be downloaded (noted in folder name)"}
SHORT = {"needle": "N", "lidocaine": "L"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 8,
    "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True, "legend.frameon": False,
    "savefig.dpi": 200, "savefig.bbox": "tight"})


def local(t):
    return pd.to_datetime(np.asarray(t, float), unit="s", utc=True).tz_convert(TZ).tz_localize(None)


def rolling_rmssd(beats, contig, t0, t1, step=5):
    g = np.arange(t0, t1, step)
    return g, np.array([H.prv(beats, contig, x - 15, x + 15, min_pairs=8)["rmssd"] for x in g])


def load_session(sid):
    sig = B.load_e4(sid) if sid in B.E4_SUBJECTS else B.load_embrace(sid)
    beats, contig = H.clean_ibi(H.beats_e4(sid) if sid in B.E4_SUBJECTS else H.beats_embrace(sid))
    ev = B.load_events(sid, sig["eda"].index.min())
    return sig, beats, contig, ev


def overview(sid):
    sig, beats, contig, ev = load_session(sid)
    rec0, rec1 = sig["eda"].index.min(), sig["eda"].index.max()
    t0, t1 = max(rec0, ev.t_utc.min() - 900), min(rec1, ev.t_utc.max() + 900)   # +-15 min around events
    ev = ev.reset_index(drop=True)
    ev["n"] = np.arange(1, len(ev) + 1)
    proc = ev[ev.category == "procedural"]
    post = ev[ev.category == "post"]
    rg, rm = rolling_rmssd(beats, contig, t0, t1)
    panels = [("eda", "EDA (µS)"), ("hr", "Heart rate (bpm)"), ("rmssd", "PPG RMSSD (ms), 30 s"),
              ("bvp_amp", "BVP amplitude (a.u.)"), ("temp", "Skin temp. (°C)"), ("motion", "Wrist motion (g)")]

    fig = plt.figure(figsize=(11, 12.5))
    gs = fig.add_gridspec(len(panels) + 3, 5, height_ratios=[1.3] + [1] * len(panels) + [0.45, 1.6],
                          width_ratios=[1, 1, 1, 1, 0.9], hspace=0.35, wspace=0.35)
    axs = [fig.add_subplot(gs[0, :4])]
    for i in range(len(panels)):
        axs.append(fig.add_subplot(gs[i + 1, :4], sharex=axs[0]))
    lt = lambda x: local(x)

    def phases(ax):
        if len(proc):
            ax.axvspan(lt([proc.t_utc.min()])[0], lt([proc.t_utc.max()])[0], color="#e9f1fc", lw=0, zorder=0)
        if len(post):
            ax.axvspan(lt([post.t_utc.min()])[0], lt([t1])[0], color=WASH, lw=0, zorder=0)
        for _, e in ev.iterrows():
            ax.axvline(lt([e.t_utc])[0], color=MUTED if e.category == "procedural" else GRID,
                       lw=0.6, ls="-" if e.category == "procedural" else "--", zorder=0.5)

    ax = axs[0]
    phases(ax)
    for k, (col, c, mk, lab) in enumerate(RATERS):
        if ev[col].notna().any():
            off = pd.to_timedelta((k - 1) * 8, unit="s")
            ax.plot(lt(ev.t_utc) + off, ev[col], mk, color=c, ms=5, mec="white", mew=0.7, ls="none", label=lab)
    for _, e in ev.iterrows():
        ax.text(lt([e.t_utc])[0], 10.9, str(e.n), ha="center", va="bottom", fontsize=6.5, color=INK2)
    ax.set_ylim(-0.8, 10.6)
    ax.set_ylabel("Pain (NRS)")
    ax.legend(loc="upper left", ncol=3, bbox_to_anchor=(0, 1.32))
    pid, dev = PATIENT.get(sid, "?"), ev.device.iloc[0]
    day = local([rec0])[0].strftime("%Y-%m-%d")
    ax.set_title(f"Session {sid} ({pid}, {dev}, {day}) — blue band: needle/lidocaine phase, "
                 f"gray band: end of procedure / recovery", loc="left", color=INK, pad=22)
    for ax, (k, lab) in zip(axs[1:], panels):
        phases(ax)
        if k == "rmssd":
            ax.plot(lt(rg), rm, color=BLUE, lw=0.9)
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
            ax.yaxis.set_minor_formatter(plt.matplotlib.ticker.NullFormatter())
        else:
            s = sig[k]
            s = s[(s.index >= t0) & (s.index <= t1)]
            if k == "hr":
                s = s.rolling(5, center=True, min_periods=1).median()
            ax.plot(lt(s.index), s.values, color=BLUE if k != "motion" else INK2, lw=0.8)
        ax.set_ylabel(lab)
        if k == "eda" and np.nanmax(sig["eda"].values) < 0.05:
            ax.text(0.01, 0.85, "EDA near sensor floor", transform=ax.transAxes, color=ORANGE, fontsize=7)
    for ax in axs[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axs[-1].set_xlabel("Local time")
    axs[0].set_xlim(lt([t0])[0], lt([t1])[0])

    # event-locked traces (needle / lidocaine)
    norm = Normalize(0, 10)
    rel = np.arange(-60, 121)
    for j, (k, lab) in enumerate([("eda", "EDA change (µS)"), ("hr", "HR change (bpm)"),
                                  ("bvp_amp", "BVP amp. change"), ("motion", "Motion change (g)")]):
        ax = fig.add_subplot(gs[-1, j])
        s = sig[k]
        for _, e in proc.iterrows():
            w = s.reindex((e.t_utc + rel).astype(int))
            if w.notna().sum() < 60:
                continue
            w = w.to_numpy() - np.nanmean(w.to_numpy()[:60])
            c = PAIN_CMAP(norm(e.pain_patient)) if pd.notna(e.pain_patient) else MUTED
            ax.plot(rel, w, color=c, lw=0.9, alpha=0.9)
        ax.axvline(0, color=INK2, lw=0.7)
        ax.set_title(lab, loc="left", fontsize=7.5, color=INK)
        ax.set_xlabel("s from event")
    cax = fig.add_subplot(gs[-1, 4])
    cax.set_axis_off()
    sm = plt.cm.ScalarMappable(norm=norm, cmap=PAIN_CMAP)
    cb = fig.colorbar(sm, ax=cax, location="left", fraction=0.12, pad=0.0)
    cb.set_label("Patient pain (NRS)", fontsize=7)
    # event list
    lines = []
    for _, e in ev.iterrows():
        sc = "/".join("–" if pd.isna(e[c]) else f"{e[c]:g}" for c, *_ in RATERS)
        lines.append(f"{e.n:>2} {lt([e.t_utc])[0]:%H:%M:%S} {e.event[:22]} ({sc})")
    tax = fig.add_subplot(gs[1:4, 4])
    tax.set_axis_off()
    tax.text(0, 1, "Events (patient/nurse/physician)\n" + "\n".join(lines), va="top", ha="left",
             fontsize=6, family="monospace", color=INK, transform=tax.transAxes)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"session_{sid:02d}.png")
    plt.close(fig)

    def avail(k):
        s = sig[k][(sig[k].index >= ev.t_utc.min() - 180) & (sig[k].index <= ev.t_utc.max() + 90)]
        span = ev.t_utc.max() - ev.t_utc.min() + 270
        return round(min(1, s.notna().sum() / span), 2)
    rm_ev = [H.prv(beats, contig, t - 30, t + 90)["rmssd"] for t in ev.t_utc]
    inside = ((ev.t_utc >= rec0) & (ev.t_utc <= rec1)).sum()
    return {"session": sid, "patient": pid, "device": dev, "date_local": day,
            "recording_start_local": f"{local([rec0])[0]:%H:%M}", "recording_end_local": f"{local([rec1])[0]:%H:%M}",
            "recording_min": round((rec1 - rec0) / 60), "assessments_in_sheet": len(ev),
            "assessments_in_recording": int(inside), "clinician_ratings": bool(ev.pain_nurse.notna().any()),
            "eda_coverage": avail("eda"), "hr_coverage": avail("hr"), "acc_coverage": avail("motion"),
            "rmssd_available_events": f"{np.mean([not np.isnan(x) for x in rm_ev]):.0%}",
            "eda_median_uS": round(float(np.nanmedian(sig['eda'].values)), 4),
            "included": True, "note": ""}


def excluded_row(sid):
    xls = [f for f in B.subject_dir(sid).glob("*.xlsx") if not f.name.startswith("~")][0]
    df = pd.read_excel(xls)
    pc = [c for c in df.columns if "pain" in str(c).lower()]
    n = int(df[pc].apply(pd.to_numeric, errors="coerce").notna().any(axis=1).sum())
    return {"session": sid, "patient": "see note", "device": "E4" if sid < 16 else "EmbracePlus",
            "date_local": B.study_date(sid).isoformat(), "assessments_in_sheet": n,
            "assessments_in_recording": 0, "clinician_ratings": any("nurse" in str(c).lower() for c in pc),
            "included": False, "note": EXCLUDED[sid]}


def main():
    rows = []
    for sid in range(1, 25):
        if sid in EXCLUDED:
            rows.append(excluded_row(sid))
            print(f"session {sid:2d}: excluded ({EXCLUDED[sid]})", flush=True)
            continue
        rows.append(overview(sid))
        print(f"session {sid:2d}: figure written", flush=True)
    inv = pd.DataFrame(rows)
    inv.to_csv(OUT / "session_inventory.csv", index=False)
    print(inv.to_string(index=False))


if __name__ == "__main__":
    main()
