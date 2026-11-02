"""Explainability: SHAP values from LightGBM and adverse action reason codes."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import config


def shap_values(model, X):
    """Per-applicant SHAP values (log-odds contributions) via LightGBM's built-in TreeSHAP.
    Positive = pushes this applicant's default risk up."""
    contrib = model.predict(X, pred_contrib=True)
    return pd.DataFrame(contrib[:, :-1], columns=X.columns, index=X.index)   # last column = base value


def plot_importance(shap_df, path, top=15):
    imp = shap_df.abs().mean().sort_values().tail(top)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(imp.index, imp.values)
    ax.set(xlabel="Mean |SHAP value| (log-odds)", title="What drives the LightGBM default model")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def reason_codes(shap_df, n_reasons=4):
    """The features that raised each applicant's risk the most, written as adverse action reasons
    (ECOA / Regulation B requires lenders to tell declined applicants the principal reasons)."""
    rows = []
    for _, row in shap_df.iterrows():
        top = row[row > 0].sort_values(ascending=False).head(n_reasons)
        rows.append({f"reason_{i + 1}": config.REASONS.get(f, f) for i, f in enumerate(top.index)})
    return pd.DataFrame(rows, index=shap_df.index)
