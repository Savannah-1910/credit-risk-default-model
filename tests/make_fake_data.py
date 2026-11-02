"""Generate a small FAKE file in the raw Lending Club format, to check the pipeline runs end to end
before downloading the real 1.6 GB dataset. The numbers it produces mean nothing.

    .venv\\Scripts\\python tests\\make_fake_data.py data\\fake\\fake_loans.csv.gz
    .venv\\Scripts\\python -m src.pipeline --raw data\\fake\\fake_loans.csv.gz --reports data\\fake\\reports
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def make(n=40_000, seed=0):
    rng = np.random.default_rng(seed)
    issue = pd.to_datetime(rng.choice(pd.date_range("2012-01-01", "2015-12-01", freq="MS"), n))
    fico_low = np.clip((rng.normal(695, 30, n) // 5) * 5, 660, 845)
    annual_inc = np.round(np.exp(rng.normal(11.1, 0.5, n)), -2)
    loan_amnt = np.round(rng.uniform(1000, 35000, n), -2)
    dti = rng.uniform(0, 40, n)
    inq = rng.poisson(0.8, n)
    revol_util = rng.uniform(0, 100, n)
    logit = (-1.8 - 0.025 * (fico_low - 700) + 0.03 * (dti - 18) + 0.25 * inq
             + 1.5 * (loan_amnt / annual_inc - 0.2) + 0.008 * (revol_util - 50))
    bad = rng.random(n) < 1 / (1 + np.exp(-logit))

    grade_idx = np.clip(((760 - fico_low) / 100 * 35 + rng.normal(0, 4, n)).astype(int), 0, 34)
    sub_grade = [f"{'ABCDEFG'[g // 5]}{g % 5 + 1}" for g in grade_idx]
    rate = 0.06 + grade_idx / 34 * 0.20
    total_pymnt = np.where(bad, loan_amnt * rng.uniform(0.1, 0.8, n), loan_amnt * (1 + rate * 1.6))
    status = np.where(bad, "Charged Off", "Fully Paid")
    status = np.where(rng.random(n) < 0.03, "Current", status)
    earliest = issue - pd.to_timedelta(rng.uniform(2, 30, n) * 365, unit="D")

    return pd.DataFrame({
        "id": np.arange(n),
        "issue_d": issue.strftime("%b-%Y"),
        "term": rng.choice([" 36 months", " 60 months"], n, p=[0.8, 0.2]),
        "loan_status": status,
        "loan_amnt": loan_amnt,
        "funded_amnt": loan_amnt,
        "sub_grade": sub_grade,
        "emp_length": rng.choice(["< 1 year", "1 year", "3 years", "5 years", "10+ years", None], n),
        "home_ownership": rng.choice(["RENT", "MORTGAGE", "OWN", "ANY"], n, p=[0.45, 0.45, 0.099, 0.001]),
        "annual_inc": annual_inc,
        "verification_status": rng.choice(["Verified", "Source Verified", "Not Verified"], n),
        "purpose": rng.choice(["debt_consolidation", "credit_card", "home_improvement", "other"], n),
        "dti": dti.round(2),
        "delinq_2yrs": rng.poisson(0.3, n),
        "earliest_cr_line": earliest.strftime("%b-%Y"),
        "fico_range_low": fico_low,
        "fico_range_high": fico_low + 4,
        "inq_last_6mths": inq,
        "mths_since_last_delinq": np.where(rng.random(n) < 0.5, np.nan, rng.integers(1, 120, n)),
        "open_acc": rng.poisson(11, n),
        "pub_rec": rng.poisson(0.2, n),
        "revol_bal": np.round(rng.exponential(15000, n)),
        "revol_util": [f"{u:.1f}%" for u in revol_util],      # string with % like older exports
        "total_acc": rng.poisson(25, n),
        "mort_acc": rng.poisson(1.5, n),
        "pub_rec_bankruptcies": rng.binomial(1, 0.1, n),
        "total_pymnt": total_pymnt.round(2),
    })


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/fake/fake_loans.csv.gz")
    out.parent.mkdir(parents=True, exist_ok=True)
    make().to_csv(out, index=False)
    print(f"Wrote fake data to {out}")
