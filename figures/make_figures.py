"""
Figures for the JMIR revision, from event_pipeline/out/jmir/*.csv (jmir_analysis.py).
Writes PNG (300 dpi, for Word) + PDF to Revisions/JMIR_revision/figures/.
Palette: validated categorical slots 1-3 (blue, orange, aqua) = patient, nurse, physician; every
series also has its own marker and a legend.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN = Path(__file__).resolve().parent / "out" / "jmir"
FIG = ROOT / "Revisions" / "JMIR_revision" / "figures"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
RATERS = {"pain_patient": (BLUE, "o", "Patient"), "pain_nurse": (ORANGE, "s", "Nurse"),
          "pain_physician": (AQUA, "^", "Physician")}
FEATS = ["baseline (training mean/prior)", "physiological", "context", "physiological+context"]
FEAT_LAB = {"baseline (training mean/prior)": "Baseline (training mean)", "physiological": "Physiological",
            "context": "Context", "physiological+context": "Physiological + context"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.dpi": 300})


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png")
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)


def fig_distribution():
    d = pd.read_csv(IN / "score_distribution.csv")
    fig, ax = plt.subplots(figsize=(6.3, 2.3))
    w = 0.26
    for i, (r, (c, _, lab)) in enumerate(RATERS.items()):
        n = d[r].sum()
        ax.bar(d.NRS + (i - 1) * w, d[r] / n * 100, width=w - 0.03, color=c, label=f"{lab} (n = {n})")
    ax.set_xticks(range(11))
    ax.set_xlabel("Pain score (NRS 0–10)")
    ax.set_ylabel("Assessments (%)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    save(fig, "Figure1_score_distribution")


def ci_row(df, rater, feat, metric):
    x = df[(df.rater == rater) & (df.features == feat) & (df.metric == metric)].iloc[0]
    return x.estimate, x.ci_low, x.ci_high


def fig_performance():
    p = pd.read_csv(IN / "performance_ci.csv")
    p = p[p.task == "regression"]
    fig, axs = plt.subplots(1, 3, figsize=(6.9, 2.5), sharey=True, gridspec_kw={"wspace": 0.12})
    for ax, metric, xl in zip(axs, ["MAE", "R2", "within_session_spearman"],
                              ["MAE (NRS points), lower is better", "R² (pooled held-out events)",
                               "Within-patient Spearman ρ"]):
        for j, f in enumerate(FEATS):
            for i, (r, (c, mk, lab)) in enumerate(RATERS.items()):
                if metric == "within_session_spearman" and f.startswith("baseline"):
                    continue
                e, lo, hi = ci_row(p, r, f, metric)
                y = len(FEATS) - 1 - j + (1 - i) * 0.22
                ax.plot([lo, hi], [y, y], color=c, lw=1.4)
                ax.plot(e, y, mk, color=c, ms=5, mec="white", mew=0.8, label=lab if j == 1 else None)
        if metric != "MAE":
            ax.axvline(0, color=INK2, lw=0.8)
        ax.set_xlabel(xl)
        ax.grid(axis="y", visible=False)
    axs[0].set_yticks(range(len(FEATS)))
    axs[0].set_yticklabels([FEAT_LAB[f] for f in FEATS[::-1]])
    h, l = axs[1].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.55, 1.06))
    for a, s in zip(axs, "abc"):
        a.text(-0.02, 1.03, s, transform=a.transAxes, fontweight="bold", fontsize=9, ha="right", va="bottom")
    save(fig, "Figure3_model_performance")


NAMES = {"eda": "EDA", "hr": "HR", "temp": "TEMP", "bvp": "BVP amp.", "motion": "Motion"}


def pretty(f):
    for k, v in [("eda_scr_n", "EDA SCR count"), ("eda_scr_amp", "EDA SCR amplitude")]:
        if f == k:
            return v
    k = next(k for k in NAMES if f.startswith(k))
    rest = f[len(k) + 1:].replace("amp_", "")
    return f"{NAMES[k]} {rest.replace('_', ' ')}"


def fig_shap():
    s = pd.read_csv(IN / "shap_by_rater.csv")
    fig, axs = plt.subplots(1, 3, figsize=(6.9, 2.8), gridspec_kw={"wspace": 0.85})
    for ax, (r, (c, _, lab)) in zip(axs, RATERS.items()):
        x = s[s.rater == r].sort_values("share", ascending=False).head(10)[::-1]
        ax.barh(range(len(x)), x.share * 100, color=c, height=0.62)
        ax.set_yticks(range(len(x)))
        ax.set_yticklabels([pretty(f) for f in x.feature])
        ax.set_title(lab, loc="left", color=INK)
        ax.set_xlabel("Share of mean absolute SHAP (%)")
        ax.grid(axis="y", visible=False)
    save(fig, "Figure2_shap_by_rater")


def fig_thresholds():
    c = pd.read_csv(IN / "classification_ci.csv")
    c = c[c.scheme.str.startswith("binary") & (c.metric == "AUROC") & (c.approach == "classifier")]
    c["T"] = c.scheme.str.replace("binary_ge", "").astype(int)
    fig, axs = plt.subplots(1, 3, figsize=(6.9, 2.4), sharey=True, gridspec_kw={"wspace": 0.1})
    styles = {"physiological": (BLUE, "o", "Physiological"), "context": (ORANGE, "s", "Context"),
              "physiological+context": (AQUA, "^", "Physiological + context")}
    for ax, (r, (_, _, lab)) in zip(axs, RATERS.items()):
        for f, (col, mk, fl) in styles.items():
            x = c[(c.rater == r) & (c.features == f)].sort_values("T")
            off = {"physiological": -0.12, "context": 0, "physiological+context": 0.12}[f]
            ax.vlines(x["T"] + off, x.ci_low, x.ci_high, color=col, lw=1.3)
            ax.plot(x["T"] + off, x.estimate, mk, color=col, ms=5, mec="white", mew=0.8, label=fl)
        ax.axhline(0.5, color=INK2, lw=0.8)
        ax.set_title(lab, loc="left", color=INK)
        ax.set_xlabel("Threshold T (score ≥ T)")
        ax.set_xticks(sorted(c["T"].unique()))
        ax.grid(axis="x", visible=False)
    axs[0].set_ylabel("AUROC (95% CI)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08))
    save(fig, "FigureS1_threshold_sensitivity")


def fig_confusion():
    c = pd.read_csv(IN / "confusion_3class.csv")
    names = ["0–3", "4–6", "7–10"]
    fig, axs = plt.subplots(2, 3, figsize=(6.6, 4.2), gridspec_kw={"hspace": 0.55, "wspace": 0.35})
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("b", ["#f4f8fe", "#86b6ef", "#2a78d6", "#104281"])
    for i, f in enumerate(["physiological", "context"]):
        for j, (r, (_, _, lab)) in enumerate(RATERS.items()):
            ax = axs[i, j]
            m = c[(c.rater == r) & (c.features == f)].pivot(index="true", columns="pred", values="n").to_numpy()
            rec = m / m.sum(1, keepdims=True)
            ax.imshow(rec, cmap=cmap, vmin=0, vmax=1)
            for a in range(3):
                for b in range(3):
                    ax.text(b, a, f"{m[a, b]}\n{rec[a, b]:.0%}", ha="center", va="center", fontsize=6.5,
                            color="white" if rec[a, b] > 0.55 else INK)
            ax.set_xticks(range(3), names)
            ax.set_yticks(range(3), names)
            ax.grid(False)
            ax.set_title(f"{lab}: {'physiological' if f == 'physiological' else 'context'}", loc="left",
                         fontsize=7.5, color=INK)
            if j == 0:
                ax.set_ylabel("True category")
            if i == 1:
                ax.set_xlabel("Predicted category")
    save(fig, "FigureS2_confusion_3class")


def main():
    fig_distribution()
    fig_performance()
    fig_shap()
    if (IN / "classification_ci.csv").exists():
        fig_thresholds()
        fig_confusion()
    print("figures ->", FIG)


if __name__ == "__main__":
    main()
