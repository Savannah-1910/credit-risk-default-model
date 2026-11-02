"""Turn the raw Lending Club file into a modeling table: one row per resolved loan."""
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

NUMERIC_RAW = [
    "loan_amnt", "funded_amnt", "annual_inc", "dti", "delinq_2yrs", "fico_range_low",
    "fico_range_high", "inq_last_6mths", "mths_since_last_delinq", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc", "mort_acc", "pub_rec_bankruptcies", "total_pymnt",
]


def find_raw_file(path=None):
    """Return the raw file path, accepting either the .csv.gz or an unzipped .csv."""
    if path is not None:
        return Path(path)
    candidates = sorted(config.RAW_DIR.glob("accepted_2007_to_2018*"))
    if not candidates:
        raise FileNotFoundError(
            f"No Lending Club file in {config.RAW_DIR}.\n"
            "Download accepted_2007_to_2018Q4.csv.gz from "
            "https://www.kaggle.com/datasets/wordsforthewise/lending-club and put it there."
        )
    return candidates[0]


def load_raw(path):
    return pd.read_csv(path, usecols=config.RAW_COLUMNS, low_memory=False)


def _to_number(s):
    # Some exports store rates as "13.5%" strings, others as floats.
    return pd.to_numeric(s.astype(str).str.replace("%", "", regex=False).str.strip(), errors="coerce")


def build_dataset(raw):
    df = raw.copy()

    # Sample definition: 36-month loans from the chosen vintages.
    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    df = df[df["issue_d"].notna() & (df["term"].astype(str).str.strip() == config.TERM)]
    df["issue_year"] = df["issue_d"].dt.year
    df = df[df["issue_year"].isin(config.TRAIN_YEARS + [config.OOT_YEAR])]

    # Target: 1 = charged off / defaulted, 0 = fully paid. Unresolved loans are dropped.
    status = df["loan_status"].astype(str).str.strip()
    df["bad"] = np.select(
        [status.isin(config.BAD_STATUSES), status.isin(config.GOOD_STATUSES)], [1, 0], default=-1
    )
    df = df[df["bad"] >= 0].copy()

    for col in NUMERIC_RAW:
        df[col] = _to_number(df[col])

    # Engineered features, all computable at application time.
    df["fico"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
    df["loan_to_income"] = df["loan_amnt"] / df["annual_inc"].where(df["annual_inc"] > 0)
    earliest = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce")
    df["credit_history_years"] = (df["issue_d"] - earliest).dt.days / 365.25
    emp = df["emp_length"].astype(str).str.replace("< 1", "0", regex=False)
    df["emp_length_years"] = pd.to_numeric(emp.str.extract(r"(\d+)")[0], errors="coerce")
    df["dti"] = df["dti"].where(df["dti"] >= 0)
    df["home_ownership"] = df["home_ownership"].replace({"ANY": "OTHER", "NONE": "OTHER"})

    # Realized economics, used only to evaluate strategies, never as model inputs.
    funding_cost = df["funded_amnt"] * config.COST_OF_FUNDS * config.AVG_LIFE_YEARS
    df["profit"] = df["total_pymnt"] - df["funded_amnt"] - funding_cost

    df["sample"] = np.where(df["issue_year"] == config.OOT_YEAR, "oot", "train")
    keep = ["id", "issue_d", "issue_year", "sample", "bad", "sub_grade", "funded_amnt",
            "total_pymnt", "profit"] + config.FEATURES
    return df[keep].reset_index(drop=True)


def get_dataset(raw_path=None, use_cache=True, rebuild=False):
    """Load the modeling table, building it from the raw file (and caching it) when needed."""
    if use_cache and config.PROCESSED_DATA.exists() and not rebuild:
        return pd.read_parquet(config.PROCESSED_DATA)
    path = find_raw_file(raw_path)
    print(f"Reading {path.name} (the full file takes a minute or two)...")
    df = build_dataset(load_raw(path))
    if use_cache:
        config.PROCESSED_DATA.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(config.PROCESSED_DATA, index=False)
    return df


def summarize(df):
    """Loan count and bad rate by vintage: the first table any credit reviewer asks for."""
    return (df.groupby(["issue_year", "sample"])
              .agg(loans=("bad", "size"), bad_rate=("bad", "mean"),
                   avg_fico=("fico", "mean"), avg_loan=("loan_amnt", "mean"))
              .reset_index())
