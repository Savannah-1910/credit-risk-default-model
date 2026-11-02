"""End-to-end run: data -> scorecard + LightGBM -> validation -> explanations -> cutoff strategy.

Usage (from the project folder):
    .venv\\Scripts\\python -m src.pipeline             # full Lending Club file in data/raw/
    .venv\\Scripts\\python -m src.pipeline --rebuild   # re-create the cached table after editing config.py
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config, data_prep, evaluation, explain, gbm, strategy
from .scorecard import Scorecard


def subgrade_rank(sub_grade):
    """A1 -> 0 ... G5 -> 34: Lending Club's own risk ranking, used as a benchmark."""
    s = sub_grade.astype(str)
    rank = (s.str[0].map(ord) - ord("A")) * 5 + pd.to_numeric(s.str[1], errors="coerce") - 1
    return rank.fillna(rank.max() + 1).to_numpy()


def main():
    sys.stdout.reconfigure(encoding="utf-8")   # the project path contains non-ASCII characters
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", help="raw Lending Club csv(.gz); default: the file in data/raw/")
    parser.add_argument("--reports", default=str(config.REPORTS), help="output folder for tables and charts")
    parser.add_argument("--rebuild", action="store_true", help="ignore the cached processed table")
    args = parser.parse_args()

    reports = Path(args.reports)
    figures = reports / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    # ---- 1. Data -------------------------------------------------------------------------------
    df = data_prep.get_dataset(args.raw, use_cache=args.raw is None, rebuild=args.rebuild)
    data_prep.summarize(df).to_csv(reports / "data_summary.csv", index=False)

    dev = df[df["sample"] == "train"]
    oot = df[df["sample"] == "oot"].reset_index(drop=True)
    train, valid = train_test_split(dev, test_size=config.VALID_SHARE, stratify=dev["bad"],
                                    random_state=config.RANDOM_STATE)
    splits = {"train": train, "valid": valid, "oot": oot}
    for name, d in splits.items():
        print(f"{name:>5}: {len(d):>8,} loans, bad rate {d['bad'].mean():.2%}")

    # ---- 2. Champion: WoE logistic scorecard ---------------------------------------------------
    sc = Scorecard(config.NUMERIC_FEATURES, config.CATEGORICAL_FEATURES).fit(train[config.FEATURES], train["bad"])
    sc.binner.iv_table().to_csv(reports / "information_value.csv", index=False)
    sc.coef_table().to_csv(reports / "scorecard_coefficients.csv", index=False)
    sc.points_table().to_csv(reports / "scorecard_points.csv", index=False)

    # ---- 3. Challenger: LightGBM with monotone constraints --------------------------------------
    categories = gbm.learn_categories(train)
    X = {name: gbm.to_gbm_frame(d, categories) for name, d in splits.items()}
    lgbm = gbm.train_lgbm(X["train"], train["bad"], X["valid"], valid["bad"])
    print(f"LightGBM stopped at {lgbm.best_iteration_} trees")

    preds = {
        "Scorecard": {n: sc.predict_pd(d[config.FEATURES]) for n, d in splits.items()},
        "LightGBM": {n: lgbm.predict_proba(X[n])[:, 1] for n in splits},
    }

    # ---- 4. Validation -------------------------------------------------------------------------
    metrics = pd.DataFrame([
        {"model": m, "sample": s, **evaluation.metrics(splits[s]["bad"], p)}
        for m, by_sample in preds.items() for s, p in by_sample.items()
    ])
    metrics.to_csv(reports / "model_metrics.csv", index=False)
    print("\n" + metrics.pivot(index="model", columns="sample", values=["auc", "ks"]).round(3).to_string())

    oot_curves = {m: (oot["bad"], preds[m]["oot"]) for m in preds}
    evaluation.plot_roc(oot_curves, figures / "roc_oot.png")
    evaluation.plot_calibration(oot_curves, figures / "calibration_oot.png")
    deciles = {m: evaluation.decile_table(y, p) for m, (y, p) in oot_curves.items()}
    pd.concat(deciles, names=["model"]).reset_index(level=0).to_csv(reports / "decile_table_oot.csv", index=False)
    evaluation.plot_deciles(deciles, figures / "rank_ordering_oot.png")

    # ---- 5. Credit strategy on the out-of-time vintage ------------------------------------------
    risk = {
        "LightGBM": preds["LightGBM"]["oot"],
        "Scorecard": preds["Scorecard"]["oot"],
        "FICO only": -oot["fico"].fillna(oot["fico"].min()).to_numpy(),
        "LC sub-grade": subgrade_rank(oot["sub_grade"]),
    }
    curves = {name: strategy.cutoff_curve(oot, r) for name, r in risk.items()}
    pd.concat(curves, names=["strategy"]).reset_index(level=0).to_csv(reports / "cutoff_curves_oot.csv", index=False)
    strategy.plot_curves(curves, figures / "cutoff_strategy_oot.png")

    best = curves["LightGBM"].loc[curves["LightGBM"]["total_profit"].idxmax()]
    approve_all = curves["LightGBM"].iloc[-1]
    swap = strategy.swap_set(oot, risk["LightGBM"], risk["FICO only"], best["approval_rate"], "LightGBM", "FICO")
    swap.to_csv(reports / "swap_set_vs_fico.csv", index=False)

    # ---- 6. Explainability -----------------------------------------------------------------------
    sample = X["oot"].sample(min(5000, len(oot)), random_state=config.RANDOM_STATE)
    explain.plot_importance(explain.shap_values(lgbm, sample), figures / "shap_importance.png")

    declined = np.flatnonzero(~strategy.approve_mask(risk["LightGBM"], best["approval_rate"]))
    if len(declined) < 20:      # profit-max policy approves (almost) everyone: show the riskiest instead
        declined = np.argsort(risk["LightGBM"])[-20:]
    rng = np.random.default_rng(config.RANDOM_STATE)
    picked = rng.choice(declined, size=min(20, len(declined)), replace=False)
    reasons = explain.reason_codes(explain.shap_values(lgbm, X["oot"].iloc[picked]))
    reasons.insert(0, "predicted_pd", preds["LightGBM"]["oot"][picked].round(3))
    reasons.insert(1, "actually_defaulted", oot["bad"].to_numpy()[picked])
    reasons.to_csv(reports / "reason_codes_sample.csv", index_label="oot_row")

    # ---- 7. Save -----------------------------------------------------------------------------------
    config.MODELS.mkdir(exist_ok=True)
    joblib.dump({"scorecard": sc, "lightgbm": lgbm, "categories": categories}, config.MODELS / "models.joblib")

    m = metrics.set_index(["model", "sample"])
    summary = {
        "oot_auc": {k: round(m.loc[(k, "oot"), "auc"], 4) for k in preds},
        "oot_ks": {k: round(m.loc[(k, "oot"), "ks"], 4) for k in preds},
        "profit_max_approval_rate": float(best["approval_rate"]),
        "profit_at_best_cutoff": round(float(best["total_profit"])),
        "profit_if_approve_all": round(float(approve_all["total_profit"])),
        "bad_rate_at_best_cutoff": round(float(best["bad_rate"]), 4),
        "bad_rate_if_approve_all": round(float(approve_all["bad_rate"]), 4),
    }
    (reports / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n" + json.dumps(summary, indent=2))
    print(f"\nSwap set vs FICO-only at {best['approval_rate']:.0%} approval:\n{swap.round(3).to_string(index=False)}")
    print(f"\nAll outputs written to {reports}")


if __name__ == "__main__":
    main()
