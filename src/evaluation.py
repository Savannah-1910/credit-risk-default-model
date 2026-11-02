"""Model validation: discrimination (AUC / Gini / KS), calibration and rank ordering."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve


def ks_stat(y, p):
    """Kolmogorov-Smirnov: max gap between the cumulative score distributions of goods and bads."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def metrics(y, p):
    auc = roc_auc_score(y, p)
    return {
        "loans": len(y),
        "bad_rate": float(np.mean(y)),
        "avg_pd": float(np.mean(p)),
        "auc": auc,
        "gini": 2 * auc - 1,
        "ks": ks_stat(y, p),
        "brier": brier_score_loss(y, p),
    }


def decile_table(y, p, n=10):
    """Decile 1 = riskiest 10%. A good model's bad rate falls steadily from decile 1 to 10."""
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    d["decile"] = pd.qcut(d["p"].rank(method="first", ascending=False), n, labels=range(1, n + 1))
    t = d.groupby("decile", observed=True).agg(loans=("y", "size"), bads=("y", "sum"), avg_pd=("p", "mean"))
    t["bad_rate"] = t["bads"] / t["loans"]
    t["cum_bad_capture"] = t["bads"].cumsum() / t["bads"].sum()
    return t.reset_index()


def plot_roc(curves, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    for label, (y, p) in curves.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=f"{label} (AUC {roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve - out-of-time sample")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_calibration(curves, path, n=10):
    """Predicted vs actual bad rate by decile. Points on the diagonal = PDs can be used as-is
    for pricing and loss forecasting."""
    fig, ax = plt.subplots(figsize=(6, 5))
    top = 0
    for label, (y, p) in curves.items():
        t = decile_table(y, p, n)
        ax.plot(t["avg_pd"], t["bad_rate"], marker="o", label=label)
        top = max(top, t["avg_pd"].max(), t["bad_rate"].max())
    ax.plot([0, top], [0, top], color="grey", linestyle="--", linewidth=1)
    ax.set(xlabel="Predicted PD", ylabel="Actual bad rate", title="Calibration by decile - out-of-time sample")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_deciles(tables, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    width = 0.8 / len(tables)
    for i, (label, t) in enumerate(tables.items()):
        ax.bar(t["decile"].astype(int) + (i - (len(tables) - 1) / 2) * width, t["bad_rate"], width, label=label)
    ax.set(xlabel="Risk decile (1 = riskiest)", ylabel="Actual bad rate", title="Rank ordering - out-of-time sample")
    ax.set_xticks(range(1, 11))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
