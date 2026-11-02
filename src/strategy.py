"""Credit strategy: turn risk rankings into approve / decline decisions and measure the money."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def cutoff_curve(df, risk, rates=None):
    """For each approval rate, approve the safest applicants by `risk` (higher = riskier) and
    measure the realized bad rate and profit of the approved book."""
    rates = np.round(np.arange(0.05, 1.0001, 0.01), 2) if rates is None else rates
    order = np.argsort(np.asarray(risk), kind="stable")
    cum_bad = np.cumsum(df["bad"].to_numpy()[order])
    cum_profit = np.cumsum(df["profit"].to_numpy()[order])
    rows = []
    for r in rates:
        k = max(int(round(r * len(df))), 1)
        rows.append({"approval_rate": r, "approved": k, "bad_rate": cum_bad[k - 1] / k,
                     "total_profit": cum_profit[k - 1], "profit_per_loan": cum_profit[k - 1] / k})
    return pd.DataFrame(rows)


def approve_mask(risk, rate):
    risk = np.asarray(risk)
    mask = np.zeros(len(risk), dtype=bool)
    mask[np.argsort(risk, kind="stable")[:int(round(rate * len(risk)))]] = True
    return mask


def swap_set(df, risk_new, risk_old, rate, new_name, old_name):
    """Compare two strategies at the same approval rate. The decision only changes for the
    swap-in / swap-out groups, so their bad rates and profit are what the new model is worth."""
    new, old = approve_mask(risk_new, rate), approve_mask(risk_old, rate)
    groups = {
        "approved by both": new & old,
        f"swap-in (only {new_name} approves)": new & ~old,
        f"swap-out (only {old_name} approves)": ~new & old,
        "declined by both": ~new & ~old,
    }
    rows = []
    for name, m in groups.items():
        sub = df[m]
        rows.append({"group": name, "loans": int(m.sum()),
                     "bad_rate": sub["bad"].mean() if len(sub) else np.nan,
                     "profit_per_loan": sub["profit"].mean() if len(sub) else np.nan,
                     "total_profit": sub["profit"].sum()})
    return pd.DataFrame(rows)


def plot_curves(curves, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for label, c in curves.items():
        ax1.plot(c["approval_rate"], c["total_profit"] / 1e6, label=label)
        ax2.plot(c["approval_rate"], c["bad_rate"], label=label)
    ax1.set(xlabel="Approval rate", ylabel="Total realized profit ($M)", title="Profit by approval cutoff")
    ax2.set(xlabel="Approval rate", ylabel="Bad rate of approved loans", title="Bad rate by approval cutoff")
    ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    for ax in (ax1, ax2):
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
