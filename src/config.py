"""Paths, sample definition, feature lists and business assumptions shared by the pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RAW_DATA = RAW_DIR / "accepted_2007_to_2018Q4.csv.gz"
PROCESSED_DATA = ROOT / "data" / "processed" / "loans.parquet"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"

RANDOM_STATE = 42

# Sample: 36-month loans issued 2012-2015. The data ends in Dec 2018, so these loans have
# (almost) all reached maturity and their outcome is known. Later vintages would only contain
# loans that already defaulted or prepaid early, which biases the bad rate.
TERM = "36 months"
TRAIN_YEARS = [2012, 2013, 2014]
OOT_YEAR = 2015          # out-of-time test: a later vintage the model never sees in training
VALID_SHARE = 0.2        # random in-time holdout carved out of the training years

BAD_STATUSES = {"Charged Off", "Default", "Does not meet the credit policy. Status:Charged Off"}
GOOD_STATUSES = {"Fully Paid", "Does not meet the credit policy. Status:Fully Paid"}
# Any other status (Current, Late, In Grace Period) is unresolved and dropped.

# Columns read from the raw file. Payment history, later FICO pulls, recoveries etc. are never
# loaded, which rules out most target leakage up front.
RAW_COLUMNS = [
    "id", "issue_d", "term", "loan_status", "loan_amnt", "funded_amnt", "sub_grade",
    "emp_length", "home_ownership", "annual_inc", "verification_status", "purpose", "dti",
    "delinq_2yrs", "earliest_cr_line", "fico_range_low", "fico_range_high", "inq_last_6mths",
    "mths_since_last_delinq", "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
    "mort_acc", "pub_rec_bankruptcies", "total_pymnt",
]

# Model inputs: only information available at the time of application.
NUMERIC_FEATURES = [
    "fico", "dti", "annual_inc", "loan_amnt", "loan_to_income", "revol_util", "revol_bal",
    "inq_last_6mths", "delinq_2yrs", "mths_since_last_delinq", "pub_rec", "pub_rec_bankruptcies",
    "open_acc", "total_acc", "mort_acc", "credit_history_years", "emp_length_years",
]
CATEGORICAL_FEATURES = ["home_ownership", "verification_status", "purpose"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Deliberately NOT used as features:
#   grade, sub_grade, int_rate, installment -> Lending Club's own risk pricing. We build our own
#                                             risk view; sub_grade is only used as a benchmark.
#   addr_state, zip_code                    -> geography can proxy for protected classes
#                                             (fair lending / ECOA disparate impact risk).
#   total_pymnt, recoveries, last_*         -> only known after the loan is booked (leakage).
#                                             total_pymnt is used to measure realized profit only.

# Expected direction of each feature on default risk (+1 = riskier as it grows, -1 = safer).
# LightGBM is forced to respect these so its decisions stay explainable to a credit committee.
MONOTONE = {
    "fico": -1, "annual_inc": -1, "dti": 1, "revol_util": 1, "inq_last_6mths": 1,
    "delinq_2yrs": 1, "pub_rec_bankruptcies": 1, "loan_to_income": 1,
}

# Plain-English adverse action reasons, used to turn SHAP values into reason codes.
REASONS = {
    "fico": "Credit score too low",
    "dti": "Debt-to-income ratio too high",
    "annual_inc": "Income insufficient for amount of credit requested",
    "loan_amnt": "Requested loan amount too high",
    "loan_to_income": "Loan amount too high relative to income",
    "revol_util": "Revolving credit utilization too high",
    "revol_bal": "Revolving balances too high",
    "inq_last_6mths": "Too many recent credit inquiries",
    "delinq_2yrs": "Recent delinquency on credit file",
    "mths_since_last_delinq": "Time since last delinquency too short",
    "pub_rec": "Derogatory public record",
    "pub_rec_bankruptcies": "Bankruptcy on credit file",
    "open_acc": "Number of open credit accounts",
    "total_acc": "Limited credit experience",
    "mort_acc": "Limited installment / mortgage history",
    "credit_history_years": "Length of credit history too short",
    "emp_length_years": "Length of employment too short",
    "home_ownership": "Housing status",
    "verification_status": "Income could not be verified",
    "purpose": "Loan purpose",
}

# Profit assumptions for the cutoff analysis. Realized profit per loan =
#   cash collected - principal funded - funding cost (cost of funds x average balance x life).
COST_OF_FUNDS = 0.03      # annual rate the lender pays on its own borrowing
AVG_LIFE_YEARS = 1.5      # an amortizing 3-year loan carries about half its principal on average
