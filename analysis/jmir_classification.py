"""
Revised classification analysis for JMIR, addressing threshold justification.

INCLUDES ≥5 threshold (previously missing) in threshold sensitivity analysis.

Schemes:
  binary_geT for T in [3,4,5,6,7]  (full threshold sensitivity)
  3class_clin    0-3 / 4-6 / 7-10
  3class_serlin  0-4 / 5-6 / 7-10
  4class         0 / 1-3 / 4-6 / 7-10

Primary algorithm: logistic regression (C ∈ {0.01, 0.1, 1}, nested)
Secondary: RF, XGBoost (robustness check)

Outputs: out/jmir_revised/classification_ci.csv, classification_permutation.csv, confusion_3class.csv
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from data_loading import load, feature_sets as get_feature_sets
from run_stats import class_metrics, patient_bootstrap

warnings.filterwarnings("ignore")
OUT = Path(__file__).parents[1] / "out" / "jmir_revised"
OUT.mkdir(exist_ok=True, parents=True)
RATERS = ["pain_patient", "pain_nurse", "pain_physician"]
SCHEMES = {**{f"binary_ge{t}": [t] for t in [3, 4, 5, 6, 7]},  # INCLUDES ≥5
           "3class_clin": [4, 7], "3class_serlin": [5, 7], "4class": [1e-6, 4, 7]}
LABELS = {"3class_clin": "0–3 / 4–6 / 7–10", "3class_serlin": "0–4 / 5–6 / 7–10",
          "4class": "0 / 1–3 / 4–6 / 7–10", **{f"binary_ge{t}": f"≥{t} vs <{t}" for t in [3, 4, 5, 6, 7]}}
MIN_CLASS = 10


def imp():
    return SimpleImputer(strategy="median")


def logreg_candidates():
    return {f"logreg(C={v})": make_pipeline(imp(), StandardScaler(), LogisticRegression(
        C=v, class_weight="balanced", max_iter=3000)) for v in [0.01, 0.1, 1]}


def rf_xgb_candidates():
    c = {f"rf(min_leaf={l})": make_pipeline(imp(), RandomForestClassifier(
        n_estimators=200, min_samples_leaf=l, max_features=0.33, class_weight="balanced",
        random_state=0, n_jobs=1)) for l in [3, 10]}
    c.update({f"xgb(depth={k})": XGBClassifier(n_estimators=200, max_depth=k, learning_rate=0.05,
                                                subsample=0.8, colsample_bytree=0.7, reg_lambda=5,
                                                random_state=0, n_jobs=1) for k in [2, 3]})
    return c


def all_candidates():
    return {**logreg_candidates(), **rf_xgb_candidates()}


def fit_proba(model, Xtr, ytr, Xte, k):
    present = np.unique(ytr)
    P = np.zeros((len(Xte), k))
    if len(present) == 1:
        P[:, present[0]] = 1
        return P
    remap = {c: i for i, c in enumerate(present)}
    m = clone(model)
    if isinstance(m, XGBClassifier):
        w = pd.Series(ytr).map(len(ytr) / (len(present) * pd.Series(ytr).value_counts())).to_numpy()
        m.fit(Xtr, np.vectorize(remap.get)(ytr), sample_weight=w)
    else:
        m.fit(Xtr, np.vectorize(remap.get)(ytr))
    P[:, present] = m.predict_proba(Xte)
    return P


def sel_score(y, P, k):
    m = class_metrics(y, P.argmax(1), k, proba=P)
    return m["AUROC"] if k == 2 else m["balanced_acc"]


def outer(X, y, g, tr, te, k, cands):
    best, bs = None, -np.inf
    for name, mdl in cands.items():
        P = np.zeros((len(tr), k))
        for itr, ite in GroupKFold(4).split(X[tr], y[tr], g[tr]):
            P[ite] = fit_proba(mdl, X[tr][itr], y[tr][itr], X[tr][ite], k)
        s = sel_score(y[tr], P, k)
        if s > bs:
            best, bs = name, s
    return te, fit_proba(cands[best], X[tr], y[tr], X[te], k), best


def metrics(y, cls, k, P=None, score=None, idx=None):
    idx = np.arange(len(y)) if idx is None else idx
    return class_metrics(y[idx], cls[idx], k, None if P is None else P[idx],
                         None if score is None else score[idx])


def ci_rows(base, y, cls, k, P, boots):
    est = metrics(y, cls, k, P)
    bt = pd.DataFrame([metrics(y, cls, k, P, idx=i) for i in boots])
    for m, v in est.items():
        if m == "accuracy":
            continue
        yield {**base, "metric": m, "estimate": v, "ci_low": np.nanpercentile(bt[m], 2.5),
               "ci_high": np.nanpercentile(bt[m], 97.5)}


def perm_stat(X, y, g, k, seed):
    yp = np.random.default_rng(seed).permutation(y)
    P = np.zeros((len(y), k))
    for tr, te in LeaveOneGroupOut().split(X, yp, g):
        P[te] = fit_proba(logreg_candidates()["logreg(C=0.1)"], X[tr], yp[tr], X[te], k)
    return sel_score(yp, P, k)


def main():
    d = load()
    fs = get_feature_sets(d)
    rows, perm, conf = [], [], []

    for rater in RATERS:
        dd = d[d[rater].notna()].reset_index(drop=True)
        s, g = dd[rater].to_numpy(float), dd.patient.to_numpy()
        boots = list(patient_bootstrap(g, np.random.default_rng(1)))

        for scheme, cuts in SCHEMES.items():
            y = np.digitize(s, cuts)
            k = len(cuts) + 1
            counts = np.bincount(y, minlength=k)
            if counts.min() < MIN_CLASS:
                print(f"  skip {rater} {scheme}: class counts {counts.tolist()}", flush=True)
                continue
            base = {"rater": rater, "scheme": scheme, "classes": LABELS[scheme],
                    "class_counts": "/".join(map(str, counts))}

            # prior baseline
            Pb = np.zeros((len(y), k))
            for tr, te in LeaveOneGroupOut().split(dd, y, g):
                Pb[te] = np.bincount(y[tr], minlength=k) / len(tr)
            rows += ci_rows({**base, "features": "baseline (training prior)", "approach": "prior"},
                            y, Pb.argmax(1), k, Pb, boots)

            for lab in ["physio_no_motion", "context", "physio_no_motion+context"]:
                X = dd[fs[lab]].to_numpy(float)
                cands = all_candidates()
                folds = Parallel(n_jobs=-1)(delayed(outer)(X, y, g, tr, te, k, cands)
                                            for tr, te in LeaveOneGroupOut().split(X, y, g))
                P = np.zeros((len(y), k))
                for te, pp, _ in folds:
                    P[te] = pp
                rows += ci_rows({**base, "features": lab, "approach": "classifier"},
                                y, P.argmax(1), k, P, boots)

                if scheme == "3class_clin":
                    cm = pd.crosstab(pd.Series(y, name="true"), pd.Series(P.argmax(1), name="pred"))
                    cm = cm.reindex(index=range(k), columns=range(k), fill_value=0)
                    for t_ in range(k):
                        for p_ in range(k):
                            conf.append({"rater": rater, "features": lab, "true": t_, "pred": p_,
                                         "n": int(cm.loc[t_, p_])})
            print(f"  {rater} {scheme} done", flush=True)

        # permutation tests
        X = dd[fs["physio_no_motion"]].to_numpy(float)
        for scheme in ["binary_ge4", "binary_ge5", "3class_clin"]:  # ADDED binary_ge5
            cuts = SCHEMES[scheme]
            y, k = np.digitize(s, cuts), len(cuts) + 1
            if np.bincount(y, minlength=k).min() < MIN_CLASS:
                continue
            Pobs = np.zeros((len(y), k))
            for tr, te in LeaveOneGroupOut().split(X, y, g):
                Pobs[te] = fit_proba(logreg_candidates()["logreg(C=0.1)"], X[tr], y[tr], X[te], k)
            obs = sel_score(y, Pobs, k)
            null = np.array(Parallel(n_jobs=-1)(delayed(perm_stat)(X, y, g, k, i) for i in range(1000)))
            perm.append({"rater": rater, "scheme": scheme, "metric": "AUROC" if k == 2 else "balanced_acc",
                         "observed": obs, "null_mean": null.mean(), "null_95pct": np.percentile(null, 95),
                         "p_perm": (np.sum(null >= obs) + 1) / (len(null) + 1)})
            print(f"  {rater} {scheme} permutation p = {perm[-1]['p_perm']:.4f}", flush=True)

    pd.DataFrame(rows).to_csv(OUT / "classification_ci.csv", index=False)
    pd.DataFrame(perm).to_csv(OUT / "classification_permutation.csv", index=False)
    pd.DataFrame(conf).to_csv(OUT / "confusion_3class.csv", index=False)
    print("done")


if __name__ == "__main__":
    main()
