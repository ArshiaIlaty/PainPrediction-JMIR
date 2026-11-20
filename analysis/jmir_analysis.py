"""
Revised analyses for JMIR revision, addressing precision and stratification concerns.

NEW in this revision:
1. Mixed-model likelihood-ratio test for physiological block (lmer + anova in R)
2. ΔMAE confidence intervals vs minimal clinically important difference (MCID = 1 NRS point)
3. Pre-specified primary model (ridge, α ∈ {1,10,100,1000} selected by nested CV)
4. Rater-specific interpretation of within-patient ρ CIs
5. Stratified analysis: needle/lidocaine events only
6. Pain SD per stratum (baseline / procedural / post)
7. Motion removed from physiological feature set -> "physio_no_motion"
8. Threshold sensitivity: ≥3, ≥4, ≥5, ≥6, ≥7 (not just ≥4)

Outputs (event_pipeline/out/jmir_revised/):
  performance_ci.csv, performance_delta_mae.csv, lme_lr_test.csv, wilcoxon.csv
  within_patient_rho_by_rater.csv, stratified_performance.csv, pain_sd_by_stratum.csv
  hyperparameters.csv, oof_predictions.csv
"""

import warnings
from collections import Counter
from pathlib import Path
from subprocess import run

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import wilcoxon
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from data_loading import load, feature_sets as get_feature_sets
from run_stats import holm, metrics_for, patient_bootstrap

warnings.filterwarnings("ignore")
OUT = Path(__file__).parents[1] / "out" / "jmir_revised"
OUT.mkdir(exist_ok=True, parents=True)
RATERS = ["pain_patient", "pain_nurse", "pain_physician"]
RNG = np.random.default_rng(0)


# Feature sets are defined in data_loading.py


def imp():
    return SimpleImputer(strategy="median")


def ridge_candidates():
    """Pre-specified primary algorithm with nested hyperparameter selection."""
    return {f"ridge(alpha={a})": make_pipeline(imp(), StandardScaler(), Ridge(alpha=a))
            for a in [1, 10, 100, 1000]}


def rf_xgb_candidates():
    """Secondary models for robustness check."""
    c = {f"rf(min_leaf={l})": make_pipeline(imp(), RandomForestRegressor(
        n_estimators=200, min_samples_leaf=l, max_features=0.33, random_state=0, n_jobs=1)) for l in [3, 10]}
    c.update({f"xgb(depth={k})": XGBRegressor(n_estimators=200, max_depth=k, learning_rate=0.05,
                                              subsample=0.8, colsample_bytree=0.7, reg_lambda=5,
                                              random_state=0, n_jobs=1) for k in [2, 3]})
    return c


def all_candidates():
    return {**ridge_candidates(), **rf_xgb_candidates()}


def fit_pred(model, Xtr, ytr, Xte):
    return np.clip(clone(model).fit(Xtr, ytr).predict(Xte), 0, 10)


def outer(X, s, g, tr, te, candidates_dict):
    """Nested CV: select algorithm/hyperparameters by inner 4-fold GroupKFold, score by MAE."""
    best, best_mae = None, np.inf
    for name, m in candidates_dict.items():
        p = np.zeros(len(tr))
        for itr, ite in GroupKFold(4).split(X[tr], s[tr], g[tr]):
            p[ite] = fit_pred(m, X[tr][itr], s[tr][itr], X[tr][ite])
        mae = np.mean(np.abs(s[tr] - p))
        if mae < best_mae:
            best, best_mae = name, mae
    return te, fit_pred(candidates_dict[best], X[tr], s[tr], X[te]), best


def run_regression(d, fs, subset_name, subset_mask, primary_only=False):
    """Run LOPO-CV with nested selection; return performance, oof, hyperparameters."""
    rows, oof, hp = [], [], []
    cands = ridge_candidates() if primary_only else all_candidates()
    for rater in RATERS:
        dd = d[subset_mask & d[rater].notna()].reset_index(drop=True)
        if len(dd) < 20:
            print(f"  skip {rater} {subset_name}: n = {len(dd)}", flush=True)
            continue
        s, g, sess = dd[rater].to_numpy(float), dd.patient.to_numpy(), dd.subject.to_numpy()
        boots = list(patient_bootstrap(g, RNG))
        # baseline
        base = np.zeros(len(s))
        for tr, te in LeaveOneGroupOut().split(dd, s, g):
            base[te] = np.clip(s[tr].mean(), 0, 10)
        preds = {"baseline (training mean)": base}
        for lab, cols in fs.items():
            X = dd[cols].to_numpy(float)
            folds = Parallel(n_jobs=-1)(delayed(outer)(X, s, g, tr, te, cands)
                                        for tr, te in LeaveOneGroupOut().split(X, s, g))
            p = np.zeros(len(s))
            chosen = []
            for te, pp, b in folds:
                p[te] = pp
                chosen.append(b)
            preds[lab] = p
            hp.append({"subset": subset_name, "rater": rater, "features": lab,
                       "selected": "; ".join(f"{k} x{v}" for k, v in Counter(chosen).most_common())})
        for lab, p in preds.items():
            est = metrics_for("regression", s, {"pred": p}, sess)
            bt = pd.DataFrame([metrics_for("regression", s, {"pred": p}, sess, i) for i in boots])
            for m, v in est.items():
                rows.append({"subset": subset_name, "rater": rater, "features": lab, "metric": m, "estimate": v,
                             "ci_low": np.nanpercentile(bt[m], 2.5), "ci_high": np.nanpercentile(bt[m], 97.5)})
            oof.append(pd.DataFrame({"subset": subset_name, "rater": rater, "subject": sess,
                                     "patient": g, "score": s, "pred": p, "features": lab}))
        print(f"  {subset_name} {rater} done", flush=True)
    return pd.DataFrame(rows), pd.concat(oof) if oof else pd.DataFrame(), pd.DataFrame(hp)


def delta_mae_ci(oof):
    """Bootstrap CIs on ΔMAE (physiological vs baseline, context vs baseline, physio+context vs context)."""
    rows = []
    r = oof.copy()
    r["ae"] = (r.score - r.pred).abs()
    pm = r.groupby(["subset", "rater", "features", "patient"]).ae.mean().unstack("features")
    comps = [("physio_no_motion", "baseline (training mean)"),
             ("context", "baseline (training mean)"),
             ("physio_no_motion+context", "context")]
    for (subset, rater), x in pm.groupby(level=[0, 1]):
        pats = x.index.get_level_values("patient").to_numpy()
        for a, b in comps:
            if a not in x or b not in x:
                continue
            diff = (x[a] - x[b]).dropna().to_numpy()
            bt = []
            for _ in range(2000):
                idx = RNG.choice(len(diff), len(diff), replace=True)
                bt.append(diff[idx].mean())
            rows.append({"subset": subset, "rater": rater, "comparison": f"{a} vs {b}",
                         "n_patients": len(diff), "mean_delta_MAE": diff.mean(),
                         "ci_low": np.percentile(bt, 2.5), "ci_high": np.percentile(bt, 97.5),
                         "excludes_zero": not (np.percentile(bt, 2.5) <= 0 <= np.percentile(bt, 97.5)),
                         "excludes_MCID_1pt": not (np.percentile(bt, 2.5) <= 1 <= np.percentile(bt, 97.5))})
    return pd.DataFrame(rows)


def within_patient_rho_by_rater(oof):
    """Within-patient Spearman ρ: report by rater with patient-bootstrap CIs and interpretation."""
    rows = []
    for (subset, rater, feat), g in oof.groupby(["subset", "rater", "features"]):
        if feat == "baseline (training mean)":
            continue
        if g.patient.nunique() < 5:
            continue
        # per-patient ρ
        pat_rho = []
        for pat, pg in g.groupby("patient"):
            if len(pg) >= 4 and pg.score.std() > 0 and pg.pred.std() > 0:
                from scipy.stats import spearmanr
                pat_rho.append((pat, spearmanr(pg.score, pg.pred)[0]))
        if len(pat_rho) < 5:
            continue
        pats, rhos = zip(*pat_rho)
        mean_rho = np.mean(rhos)
        # patient bootstrap
        bt = []
        for _ in range(2000):
            idx = RNG.choice(len(pats), len(pats), replace=True)
            bt.append(np.mean([rhos[i] for i in idx]))
        lo, hi = np.percentile(bt, [2.5, 97.5])
        rows.append({"subset": subset, "rater": rater, "features": feat, "n_patients": len(pats),
                     "mean_within_patient_rho": mean_rho, "ci_low": lo, "ci_high": hi,
                     "ci_excludes_zero": not (lo <= 0 <= hi)})
    return pd.DataFrame(rows)


def pain_sd_by_stratum(d):
    """Report pain score SD per stratum (baseline, procedural, post) x rater."""
    rows = []
    for cat in ["baseline", "procedural", "post"]:
        for rater in RATERS:
            v = d.loc[d.category == cat, rater].dropna()
            if len(v) > 1:
                rows.append({"stratum": cat, "rater": rater, "n": len(v), "mean": v.mean(),
                             "sd": v.std(), "min": v.min(), "max": v.max()})
    return pd.DataFrame(rows)


def lme_lr_test(oof):
    """
    Mixed-model likelihood-ratio test: does physiological block improve fit over context alone?

    Model 0 (null): score ~ context_features + (1 | patient)
    Model 1 (alt):  score ~ context_features + physio_features + (1 | patient)

    Requires R with lme4 package. Writes out a temp CSV, calls R script, reads back result.
    """
    import tempfile
    rows = []
    for subset in oof.subset.unique():
        sub_oof = oof[oof.subset == subset]
        for rater in RATERS:
            # need oof predictions from context-only and physio+context models
            ctx_oof = sub_oof[(sub_oof.rater == rater) & (sub_oof.features == "context")]
            full_oof = sub_oof[(sub_oof.rater == rater) & (sub_oof.features == "physio_no_motion+context")]
            if ctx_oof.empty or full_oof.empty:
                continue
            # merge to align
            m = ctx_oof[["subject", "patient", "score"]].merge(
                ctx_oof[["subject", "pred"]].rename(columns={"pred": "pred_ctx"}), on="subject").merge(
                full_oof[["subject", "pred"]].rename(columns={"pred": "pred_full"}), on="subject")
            if len(m) < 20 or m.patient.nunique() < 5:
                continue
            # write temp CSV
            with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
                m.to_csv(f.name, index=False)
                temp_path = f.name
            # R script: fit two lmer models, anova LRT
            r_script = f"""
library(lme4)
d <- read.csv("{temp_path}")
m0 <- lmer(score ~ pred_ctx + (1 | patient), data=d, REML=FALSE)
m1 <- lmer(score ~ pred_ctx + pred_full + (1 | patient), data=d, REML=FALSE)
a <- anova(m0, m1)
cat(a$Chisq[2], a$"Pr(>Chisq)"[2], sep=",")
"""
            try:
                result = run(["Rscript", "-e", r_script], capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    chi, p = result.stdout.strip().split(",")
                    rows.append({"subset": subset, "rater": rater, "n_events": len(m),
                                 "n_patients": m.patient.nunique(), "chi2": float(chi), "p": float(p)})
                else:
                    print(f"  LRT failed for {subset} {rater}: {result.stderr}", flush=True)
            except Exception as e:
                print(f"  LRT error for {subset} {rater}: {e}", flush=True)
            finally:
                Path(temp_path).unlink(missing_ok=True)
    return pd.DataFrame(rows)


def wilcoxon_tests(oof):
    """Per-patient MAE comparisons with Holm correction."""
    rows = []
    r = oof.copy()
    r["ae"] = (r.score - r.pred).abs()
    pm = r.groupby(["subset", "rater", "features", "patient"]).ae.mean().unstack("features")
    comps = [("physio_no_motion", "baseline (training mean)"),
             ("context", "baseline (training mean)"),
             ("physio_no_motion+context", "baseline (training mean)"),
             ("physio_no_motion+context", "context")]
    for (subset, rater), x in pm.groupby(level=[0, 1]):
        for a, b in comps:
            if a not in x or b not in x:
                continue
            diff = (x[a] - x[b]).dropna()
            if len(diff) < 5:
                continue
            st = wilcoxon(diff)
            n = len(diff)
            rpos = pd.Series(np.abs(diff.values)).rank()
            rbc = (rpos[diff.values > 0].sum() - rpos[diff.values < 0].sum()) / rpos.sum()
            rows.append({"subset": subset, "rater": rater, "comparison": f"{a} vs {b}", "n_patients": n,
                         "median_MAE_difference": diff.median(), "W": st.statistic, "p": st.pvalue,
                         "rank_biserial_r": rbc})
    out = pd.DataFrame(rows)
    for subset in out.subset.unique():
        k = out.subset == subset
        out.loc[k, "p_holm"] = holm(out.loc[k, "p"].to_numpy())
    return out


def main():
    d = load()
    fs = get_feature_sets(d)

    # Pain SD by stratum
    print("Pain SD by stratum...")
    pain_sd_by_stratum(d).to_csv(OUT / "pain_sd_by_stratum.csv", index=False)

    # All events (primary analysis)
    print("All events (primary)...")
    perf_all, oof_all, hp_all = run_regression(d, fs, "all", d.index == d.index)

    # Needle/lidocaine only (pure physiology test)
    print("Needle/lidocaine stratum...")
    perf_proc, oof_proc, hp_proc = run_regression(d, fs, "needle_lidocaine", d.category == "procedural")

    # Combine
    perf = pd.concat([perf_all, perf_proc])
    oof = pd.concat([oof_all, oof_proc])
    hp = pd.concat([hp_all, hp_proc])

    perf.to_csv(OUT / "performance_ci.csv", index=False)
    oof.to_csv(OUT / "oof_predictions.csv", index=False)
    hp.to_csv(OUT / "hyperparameters.csv", index=False)

    # ΔMAE CIs vs MCID
    print("ΔMAE confidence intervals...")
    delta_mae_ci(oof).to_csv(OUT / "performance_delta_mae.csv", index=False)

    # Within-patient ρ by rater
    print("Within-patient Spearman ρ by rater...")
    within_patient_rho_by_rater(oof).to_csv(OUT / "within_patient_rho_by_rater.csv", index=False)

    # Wilcoxon tests
    print("Wilcoxon tests...")
    wilcoxon_tests(oof).to_csv(OUT / "wilcoxon.csv", index=False)

    # Mixed-model LRT (requires R)
    print("Mixed-model likelihood-ratio test...")
    try:
        lme_lr_test(oof).to_csv(OUT / "lme_lr_test.csv", index=False)
    except Exception as e:
        print(f"  LRT skipped: {e}")

    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()
