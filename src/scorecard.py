"""Traditional credit scorecard: WoE-binned features + logistic regression, scaled to points."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

MISSING = "MISSING"
OTHER = "OTHER"


class WoEBinner:
    """Bins each feature and replaces the bin with its Weight of Evidence.

    WoE = ln(share of goods in bin / share of bads in bin), so a positive WoE means safer than
    average. Numeric features get quantile bins; categorical levels rarer than `min_share` are
    pooled into OTHER. Missing values get their own bin, so e.g. "no delinquency on file" keeps
    its own signal instead of being imputed away.
    """

    def __init__(self, numeric, categorical, n_bins=10, min_share=0.02):
        self.numeric = list(numeric)
        self.categorical = list(categorical)
        self.n_bins = n_bins
        self.min_share = min_share

    def fit(self, X, y):
        y = np.asarray(y)
        self.edges_, self.levels_, self.woe_, self.iv_ = {}, {}, {}, {}
        for col in self.numeric:
            x = X[col].to_numpy(dtype=float)
            inner = np.linspace(0, 1, self.n_bins + 1)[1:-1]
            self.edges_[col] = np.unique(np.nanquantile(x, inner)) if np.isfinite(x).any() else np.array([])
        for col in self.categorical:
            share = X[col].value_counts(normalize=True)
            self.levels_[col] = set(share[share >= self.min_share].index)
        for col in self.numeric + self.categorical:
            self.woe_[col], self.iv_[col] = self._woe_table(self._bin(X, col), y)
        return self

    def _bin(self, X, col):
        if col in self.edges_:
            x = X[col].to_numpy(dtype=float)
            bins = pd.Series(np.searchsorted(self.edges_[col], x, side="right"), index=X.index, dtype=object)
            bins[np.isnan(x)] = MISSING
            return bins
        s = X[col].astype(object)
        return s.where(s.isin(self.levels_[col]), OTHER).where(s.notna(), MISSING)

    @staticmethod
    def _woe_table(bins, y):
        t = pd.DataFrame({"bin": bins.to_numpy(), "bad": y}).groupby("bin", sort=False)["bad"].agg(["count", "sum"])
        bad = t["sum"] + 0.5                    # +0.5 keeps empty cells from producing log(0)
        good = t["count"] - t["sum"] + 0.5
        dist_good, dist_bad = good / good.sum(), bad / bad.sum()
        woe = np.log(dist_good / dist_bad)
        iv = float(((dist_good - dist_bad) * woe).sum())
        return woe.to_dict(), iv

    def transform(self, X, cols=None):
        cols = cols or self.numeric + self.categorical
        # Bins never seen in training get WoE 0, i.e. treated as average risk.
        return pd.DataFrame(
            {c: self._bin(X, c).map(self.woe_[c]).fillna(0.0).astype(float) for c in cols}, index=X.index
        )

    def bin_label(self, col, b):
        if b in (MISSING, OTHER) or col not in self.edges_:
            return str(b)
        edges = self.edges_[col]
        lo = "-inf" if b == 0 else f"{edges[b - 1]:,.2f}"
        hi = "inf" if b == len(edges) else f"{edges[b]:,.2f}"
        return f"[{lo}, {hi})"

    def iv_table(self):
        """Information Value per feature. Rule of thumb: <0.02 useless, 0.1-0.3 medium, >0.3 strong."""
        return (pd.DataFrame({"feature": list(self.iv_), "iv": list(self.iv_.values())})
                  .sort_values("iv", ascending=False).reset_index(drop=True))


def _bin_order(item):
    b = item[0]
    return (1, 0, str(b)) if isinstance(b, str) else (0, int(b), "")


class Scorecard:
    """WoE logistic regression scaled to points: `base_score` at `base_odds`:1 good:bad odds,
    and every `pdo` points doubles the odds of being good."""

    def __init__(self, numeric, categorical, iv_min=0.02, base_score=600, base_odds=50, pdo=20):
        self.numeric = numeric
        self.categorical = categorical
        self.iv_min = iv_min
        self.factor = pdo / np.log(2)
        self.offset = base_score - self.factor * np.log(base_odds)

    def fit(self, X, y):
        self.binner = WoEBinner(self.numeric, self.categorical).fit(X, y)
        iv = self.binner.iv_table()
        self.features_ = iv.loc[iv["iv"] >= self.iv_min, "feature"].tolist()
        self.model = LogisticRegression(max_iter=1000).fit(self.binner.transform(X, self.features_), y)
        return self

    def predict_pd(self, X):
        return self.model.predict_proba(self.binner.transform(X, self.features_))[:, 1]

    def score(self, X):
        p = np.clip(self.predict_pd(X), 1e-6, 1 - 1e-6)
        return self.offset + self.factor * np.log((1 - p) / p)

    def coef_table(self):
        """Coefficients on WoE should all be negative (higher WoE = safer = lower PD).
        A positive sign usually means the feature is collinear with another one."""
        iv = self.binner.iv_
        return pd.DataFrame({
            "feature": self.features_,
            "coef": self.model.coef_[0],
            "iv": [iv[f] for f in self.features_],
        }).sort_values("iv", ascending=False).reset_index(drop=True)

    def points_table(self):
        """The scorecard itself: points per feature bin. An applicant's score is the sum of the
        points of the bins they fall into."""
        n = len(self.features_)
        b0 = self.model.intercept_[0]
        rows = []
        for coef, col in zip(self.model.coef_[0], self.features_):
            for b, woe in sorted(self.binner.woe_[col].items(), key=_bin_order):
                points = self.offset / n - self.factor * (coef * woe + b0 / n)
                rows.append({"feature": col, "bin": self.binner.bin_label(col, b),
                             "woe": round(woe, 4), "points": int(round(points))})
        return pd.DataFrame(rows)
