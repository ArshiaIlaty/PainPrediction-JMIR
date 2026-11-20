"""
Statistical evaluation of the event-level models: CIs, permutation tests, paired comparisons.

All predictions are out-of-fold from leave-one-PATIENT-out (13 patients).
  - 95% CIs: cluster bootstrap over patients (resample patients with replacement,
    keep all their events), applied to the fixed out-of-fold predictions.
  - Permutation tests (H0: physiology carries no information about the score):
      global    - shuffle scores across all events, refit the whole LOPO loop;
      within    - shuffle scores within each session (keeps session means), tests whether
                  physiology ranks events *within* a patient.
    Correction across the physiology family: Benjamini-Hochberg FDR and Holm.
  - Paired comparisons (same events, same folds): physio vs baseline, physio+context vs
    context, via the patient bootstrap of the metric difference.
Tasks: regression 0-10 (R2, MAE, RMSE, Spearman), binary >=4, 3-class 0-3/4-6/7-10,
4-class 0/1-3/4-6/7-10; classifiers plus "RF regression -> bins".

Usage: python event_pipeline/run_stats.py [--n-perm-linear 2000] [--n-perm-rf 300]
Writes out/stats_*.csv and out/STATS_REPORT.md
"""

import argparse
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import rankdata
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from run_experiments import OUT, RATERS, feature_sets, load

warnings.filterwarnings("ignore")
N_BOOT = 2000
TASKS = {  # name -> class thresholds (np.digitize, right-open); None = regression
    "regression": None,
    "binary_ge4": [4],
    "3class": [4, 7],
    "4class": [1e-6, 4, 7],
}
CLASS_NAMES = {"binary_ge4": ["<4", ">=4"], "3class": ["0-3", "4-6", "7-10"],
               "4class": ["0", "1-3", "4-6", "7-10"]}
FEATURES = ["physio", "context", "physio+context"]


def reg_models():
    imp = lambda: SimpleImputer(strategy="median")
    return {"mean": DummyRegressor(),
            "ridge": make_pipeline(imp(), StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 20))),
            "rf": make_pipeline(imp(), RandomForestRegressor(
                n_estimators=100, min_samples_leaf=5, max_features=0.33, random_state=0))}


def cls_models():
    imp = lambda: SimpleImputer(strategy="median")
    return {"majority": DummyClassifier(strategy="prior"),
            "logreg": make_pipeline(imp(), StandardScaler(), LogisticRegression(
                C=0.1, class_weight="balanced", max_iter=2000)),
            "rf": make_pipeline(imp(), RandomForestClassifier(
                n_estimators=100, min_samples_leaf=5, max_features=0.33,
                class_weight="balanced", random_state=0)),
            "rf_reg_binned": reg_models()["rf"]}


# ----------------------------------------------------------------------------- fitting
def oof(model, X, y, groups, n_classes=None):
    """Out-of-fold predictions. Regression -> (n,), classification -> (n, k) probabilities."""
    out = np.zeros((len(y), n_classes)) if n_classes else np.zeros(len(y))
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        m = clone(model).fit(X[tr], y[tr])
        if n_classes:
            out[np.ix_(te, m.classes_.astype(int))] = m.predict_proba(X[te])
        else:
            out[te] = m.predict(X[te])
    return out


def predict(task, mname, model, X, score, groups):
    """Returns dict with continuous prediction / class prediction / probabilities."""
    th = TASKS[task]
    if task == "regression":
        return {"pred": np.clip(oof(model, X, score, groups), 0, 10)}
    k = len(th) + 1
    if mname == "rf_reg_binned":
        p = np.clip(oof(model, X, score, groups), 0, 10)
        return {"pred": p, "cls": np.digitize(p, th), "proba": None}
    y = np.digitize(score, th)
    P = oof(model, X, y, groups, n_classes=k)
    return {"pred": None, "cls": P.argmax(1), "proba": P}


# ----------------------------------------------------------------------------- metrics
def spearman(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return np.corrcoef(rankdata(a), rankdata(b))[0, 1]


def auroc(y, s):
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def class_metrics(y, c, k, proba=None, score=None):
    cm = np.bincount(y * k + c, minlength=k * k).reshape(k, k).astype(float)
    support, predicted, tp = cm.sum(1), cm.sum(0), np.diag(cm)
    present = support > 0
    recall = np.divide(tp, support, out=np.zeros(k), where=support > 0)
    prec = np.divide(tp, predicted, out=np.zeros(k), where=predicted > 0)
    f1 = np.divide(2 * prec * recall, prec + recall, out=np.zeros(k), where=prec + recall > 0)
    w = (np.arange(k)[:, None] - np.arange(k)[None]) ** 2
    exp = np.outer(support, predicted) / cm.sum()
    qwk = 1 - (w * cm).sum() / (w * exp).sum() if (w * exp).sum() > 0 else np.nan
    m = {"balanced_acc": recall[present].mean(), "macro_F1": f1[present].mean(),
         "accuracy": tp.sum() / cm.sum(), "QW_kappa": qwk}
    if k == 2:
        s = proba[:, 1] if proba is not None else score
        m["AUROC"] = auroc(y, s)
    elif proba is not None:
        m["AUROC_macro_ovr"] = np.nanmean([auroc((y == j).astype(int), proba[:, j])
                                           for j in range(k) if present[j]])
    return m


def reg_metrics(y, p, sess_rho=None):
    ss = ((y - y.mean()) ** 2).sum()
    m = {"R2": 1 - ((y - p) ** 2).sum() / ss if ss > 0 else np.nan,
         "MAE": np.abs(y - p).mean(), "RMSE": np.sqrt(((y - p) ** 2).mean()),
         "spearman": spearman(y, p)}
    if sess_rho is not None:
        m["within_session_spearman"] = np.nanmean(sess_rho) if len(sess_rho) else np.nan
    return m


def session_rhos(y, p, sessions):
    """Per-session Spearman (sessions with >=4 events and varying scores)."""
    out = {}
    for s in np.unique(sessions):
        i = sessions == s
        if i.sum() >= 4 and np.std(y[i]) > 0:
            out[s] = spearman(y[i], p[i])
    return out


def metrics_for(task, y_score, r, sessions, idx=None):
    idx = np.arange(len(y_score)) if idx is None else idx
    ys = y_score[idx]
    if task == "regression":
        rho = session_rhos(ys, r["pred"][idx], sessions[idx])
        return reg_metrics(ys, r["pred"][idx], np.array([v for v in rho.values()]))
    k = len(TASKS[task]) + 1
    y = np.digitize(ys, TASKS[task])
    return class_metrics(y, r["cls"][idx], k,
                         None if r["proba"] is None else r["proba"][idx],
                         None if r["pred"] is None else r["pred"][idx])


def patient_bootstrap(patients, rng, n=N_BOOT):
    """Yields index arrays; each draws patients with replacement and keeps all their events."""
    ids = np.unique(patients)
    rows = {p: np.where(patients == p)[0] for p in ids}
    for _ in range(n):
        yield np.concatenate([rows[p] for p in rng.choice(ids, len(ids))])


# ----------------------------------------------------------------------------- permutation
def perm_stat(task, mname, model, X, score, groups, sessions, kind, seed):
    rng = np.random.default_rng(seed)
    s = score.copy()
    if kind == "global":
        s = rng.permutation(s)
    else:
        for v in np.unique(sessions):
            i = np.where(sessions == v)[0]
            s[i] = rng.permutation(s[i])
    r = predict(task, mname, model, X, s, groups)
    m = metrics_for(task, s, r, sessions)
    return m[primary_metric(task, kind)]


def primary_metric(task, kind="global"):
    if task == "regression":
        return "R2" if kind == "global" else "within_session_spearman"
    return "AUROC" if task == "binary_ge4" else "balanced_acc"


# ----------------------------------------------------------------------------- correction
def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return out


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[o])
    out = np.empty_like(adj)
    out[o] = np.minimum(adj, 1)
    return out


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm-linear", type=int, default=2000)
    ap.add_argument("--n-perm-rf", type=int, default=300)
    args = ap.parse_args()
    t_start = time.time()
    d = load()
    fs = feature_sets(d)
    subsets = {"all_events": d, "procedural_only": d[d.category == "procedural"]}

    runs = []   # (key, task, y, r, sessions, patients, X, model)
    for exp, dd0 in subsets.items():
        for rater in RATERS:
            dd = dd0[dd0[rater].notna()]
            score, groups = dd[rater].to_numpy(float), dd.patient.to_numpy()
            sessions = dd.subject.to_numpy()
            for task in TASKS:
                if exp == "procedural_only" and task in ("3class", "4class"):
                    continue    # almost no 0-3 / 0 events among needles
                models = reg_models() if task == "regression" else cls_models()
                for fname in FEATURES:
                    X = dd[fs[fname]].to_numpy(float)
                    for mname, model in models.items():
                        if mname in ("mean", "majority") and fname != "physio":
                            continue
                        r = predict(task, mname, model, X, score, groups)
                        key = dict(experiment=exp, rater=rater, task=task,
                                   features="-" if mname in ("mean", "majority") else fname,
                                   model=mname)
                        runs.append((key, task, score, r, sessions, groups, X, model))
    print(f"fitted {len(runs)} configurations in {time.time() - t_start:.0f}s", flush=True)

    # ---- point estimates + patient-bootstrap CIs (same resamples for every model of a
    #      given experiment x rater, so paired differences reuse the cached replicates)
    boot_idx = {}
    for key, task, score, r, sessions, groups, X, model in runs:
        bkey = (key["experiment"], key["rater"])
        if bkey not in boot_idx:
            boot_idx[bkey] = list(patient_bootstrap(groups, np.random.default_rng(1)))
    boot_all = Parallel(n_jobs=-1)(
        delayed(lambda t, s, r, se, idx: pd.DataFrame([metrics_for(t, s, r, se, i) for i in idx]))(
            task, score, r, sessions, boot_idx[(key["experiment"], key["rater"])])
        for key, task, score, r, sessions, groups, X, model in runs)
    rows, preds, boots = [], [], {}
    for (key, task, score, r, sessions, groups, X, model), bt in zip(runs, boot_all):
        est = metrics_for(task, score, r, sessions)
        boots[tuple(key.values())] = (est, bt)
        for m, v in est.items():
            lo, hi = np.nanpercentile(bt[m], [2.5, 97.5])
            rows.append({**key, "n": len(score), "metric": m, "estimate": v, "ci_low": lo,
                         "ci_high": hi})
        preds.append(pd.DataFrame({**key, "subject": sessions, "patient": groups, "score": score,
                                   "pred": r["pred"] if r["pred"] is not None else np.nan,
                                   "pred_class": r.get("cls", np.nan)}))
    ci = pd.DataFrame(rows)
    ci.to_csv(OUT / "stats_metrics_ci.csv", index=False)
    pd.concat(preds).to_csv(OUT / "stats_oof_predictions.csv", index=False)
    print(f"bootstrap done {time.time() - t_start:.0f}s", flush=True)

    # ---- paired comparisons (bootstrap of the metric difference)
    comps = []
    for (exp, rater, task, feat, mname), (e1, b1) in boots.items():
        pairs = []
        if feat == "physio":
            base = "mean" if task == "regression" else "majority"
            pairs += [("physio vs baseline", (exp, rater, task, "-", base)),
                      ("physio vs context", (exp, rater, task, "context", mname))]
        if feat == "physio+context":
            pairs.append(("physio+context vs context", (exp, rater, task, "context", mname)))
        for label, other in pairs:
            if other not in boots:
                continue
            e2, b2 = boots[other]
            metrics = ["R2", "MAE"] if task == "regression" else \
                ["AUROC", "balanced_acc"] if task == "binary_ge4" else ["balanced_acc", "QW_kappa"]
            for m in metrics:
                diff = (b1[m] - b2[m]).dropna().to_numpy()
                p = min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean()))
                comps.append({"experiment": exp, "rater": rater, "task": task, "model": mname,
                              "comparison": label, "metric": m, "difference": e1[m] - e2[m],
                              "ci_low": np.percentile(diff, 2.5),
                              "ci_high": np.percentile(diff, 97.5), "p_boot": p})
    comps = pd.DataFrame(comps)
    comps.to_csv(OUT / "stats_comparisons.csv", index=False)

    # ---- permutation tests for physiology-only models
    jobs, meta = [], []
    for key, task, score, r, sessions, groups, X, model in runs:
        if key["features"] != "physio":
            continue
        kinds = ["global"] + (["within"] if task == "regression" and key["model"] == "ridge" else [])
        for kind in kinds:
            n_perm = args.n_perm_rf if key["model"].startswith("rf") else args.n_perm_linear
            obs = metrics_for(task, score, r, sessions)[primary_metric(task, kind)]
            meta.append({**key, "null": kind, "metric": primary_metric(task, kind),
                         "observed": obs, "n_perm": n_perm})
            jobs.append([delayed(perm_stat)(task, key["model"], model, X, score, groups,
                                            sessions, kind, seed) for seed in range(n_perm)])
    flat = [j for js in jobs for j in js]
    print(f"running {len(flat)} permutation refits on all cores ...", flush=True)
    res = Parallel(n_jobs=-1, batch_size=8)(flat)
    perm, pos = [], 0
    for m, js in zip(meta, jobs):
        null = np.array(res[pos:pos + len(js)], float)
        pos += len(js)
        null = null[~np.isnan(null)]
        perm.append({**m, "null_mean": null.mean(), "null_95pct": np.percentile(null, 95),
                     "p_perm": (np.sum(null >= m["observed"]) + 1) / (len(null) + 1)})
    perm = pd.DataFrame(perm)
    perm["q_BH"] = bh(perm.p_perm)
    perm["p_Holm"] = holm(perm.p_perm)
    perm.to_csv(OUT / "stats_permutation.csv", index=False)
    write_report(ci, comps, perm)
    print(f"done in {(time.time() - t_start) / 60:.1f} min")


def fmt_ci(row, digits=2):
    return f"{row.estimate:.{digits}f} [{row.ci_low:.{digits}f}, {row.ci_high:.{digits}f}]"


def write_report(ci, comps, perm):
    L = ["# Statistical evaluation (leave-one-patient-out, 13 patients)", "",
         "Values: estimate [95% patient-bootstrap CI]. Permutation p: refit of the full LOPO loop.",
         ""]

    def table(exp, task, metrics, models):
        sub = ci[(ci.experiment == exp) & (ci.task == task)]
        L.append(f"## {exp} - {task}")
        L.append("")
        for m in metrics:
            L.append(f"**{m}**")
            L.append("")
            L.append("| features | model | " + " | ".join(r.replace("pain_", "") for r in RATERS) + " |")
            L.append("|---|---|" + "---|" * len(RATERS))
            for (f, mo), g in sub[sub.metric == m].groupby(["features", "model"], sort=False):
                if mo not in models:
                    continue
                cells = [fmt_ci(g[g.rater == r].iloc[0]) if (g.rater == r).any() else "-"
                         for r in RATERS]
                L.append(f"| {f} | {mo} | " + " | ".join(cells) + " |")
            L.append("")

    table("all_events", "regression", ["R2", "MAE", "within_session_spearman"], ["mean", "ridge", "rf"])
    table("all_events", "binary_ge4", ["AUROC", "balanced_acc"],
          ["majority", "logreg", "rf", "rf_reg_binned"])
    table("all_events", "3class", ["balanced_acc", "macro_F1", "QW_kappa"],
          ["majority", "logreg", "rf", "rf_reg_binned"])
    table("all_events", "4class", ["balanced_acc", "macro_F1", "QW_kappa"],
          ["majority", "logreg", "rf", "rf_reg_binned"])
    table("procedural_only", "regression", ["R2", "MAE", "within_session_spearman"],
          ["mean", "ridge", "rf"])
    table("procedural_only", "binary_ge4", ["AUROC", "balanced_acc"],
          ["majority", "logreg", "rf", "rf_reg_binned"])

    L += ["## Permutation tests, physiology-only models", "",
          "| experiment | rater | task | model | null | metric | observed | null mean | p | q (BH) | p (Holm) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in perm.iterrows():
        L.append(f"| {r.experiment} | {r.rater.replace('pain_', '')} | {r.task} | {r.model} | "
                 f"{r.null} | {r.metric} | {r.observed:.3f} | {r.null_mean:.3f} | {r.p_perm:.4f} | "
                 f"{r.q_BH:.3f} | {r.p_Holm:.3f} |")
    L += ["", "## Paired comparisons (difference = first minus second; MAE: negative is better)", "",
          "| experiment | rater | task | model | comparison | metric | diff [95% CI] | p |",
          "|---|---|---|---|---|---|---|---|"]
    for _, r in comps.iterrows():
        L.append(f"| {r.experiment} | {r.rater.replace('pain_', '')} | {r.task} | {r.model} | "
                 f"{r.comparison} | {r.metric} | {r.difference:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}] | "
                 f"{r.p_boot:.3f} |")
    (OUT / "STATS_REPORT.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
