# Consumer Loan Default Model & Credit Strategy

An end-to-end consumer credit risk project on real Lending Club loans: predict which personal loans
will default, explain every decision, and turn the model into an approve / decline policy that is
measured in dollars, not just AUC.

## Business question

A consumer lender wants to replace its FICO-only approval rule. Can an in-house default model
approve more good borrowers, decline more bad ones, and make more money, while still giving every
declined applicant a clear reason, as fair lending rules require?

## Approach

| Step | What | Why it matters |
|---|---|---|
| Sample | 36-month loans issued 2012-2015, resolved as Fully Paid or Charged Off | All loans have matured, so default outcomes are not biased toward early defaults |
| Leakage control | Only application-time fields are loaded; payment history, recoveries and later FICO pulls are never read | A model that sees post-booking data looks great on paper and fails in production |
| Fair lending | Geography (state, zip) excluded; Lending Club's own grade and rate excluded from inputs | Avoids proxies for protected classes and builds an independent risk view |
| Champion | WoE-binned logistic regression scaled to a points scorecard (600 points at 50:1 odds, 20 PDO) | The industry-standard, fully transparent model regulators know |
| Challenger | LightGBM with monotone constraints (e.g. higher FICO can never raise risk) | Captures non-linear effects while staying defensible |
| Validation | Train 2012-2014, out-of-time test on the 2015 vintage; AUC, Gini, KS, calibration, decile rank ordering | Tests whether the model survives a change in the borrower population |
| Explainability | TreeSHAP values turned into adverse action reason codes | ECOA / Regulation B requires principal reasons for every decline |
| Strategy | Profit and bad rate at every approval cutoff; swap-set analysis vs a FICO-only rule | The decision a lender actually makes is the cutoff, not the AUC |

## Results

306K loans for development (2012-2014, 13.2% bad rate), 283K loans for the out-of-time test
(2015, 14.9% bad rate).

**Model performance on the 2015 out-of-time vintage**

| Model | Valid AUC | OOT AUC | OOT KS | Bad rate, riskiest vs safest decile |
|---|---|---|---|---|
| WoE scorecard (8 features, IV ≥ 0.02) | 0.645 | 0.648 | 0.213 | 27.1% vs 4.5% |
| LightGBM (monotone) | 0.669 | **0.674** | **0.250** | 29.9% vs 3.5% |

- Out-of-time AUC holds up versus the in-time holdout, so the model is stable across vintages.
- Both models rank-order cleanly: bad rate falls in every decile.
- Both under-predict slightly in 2015 because the vintage was riskier than 2012-2014
  (calibration chart). In production the intercept would be recalibrated on recent data.

**Credit strategy on the 2015 vintage** (realized profit after a 3% cost of funds)

| Policy at 87% approval | Bad rate of approved | Realized profit |
|---|---|---|
| Approve everyone (100%) | 14.9% | $74.3M |
| FICO-only cutoff | 14.0% | $70.2M |
| Lending Club sub-grade cutoff | 12.8% | $77.5M |
| **LightGBM cutoff (profit-maximizing)** | **12.8%** | **$82.3M** |

- At the same 87% approval rate, LightGBM adds **$12.1M (+17%)** over a FICO-only rule. The
  swap-set analysis shows where it comes from: the 26K loans only LightGBM approves have a 16.9%
  bad rate and earn $251 each; the 26K loans only FICO approves have a 28.4% bad rate and lose
  $208 each.
- Using only public application data, the model matches the bad rate of Lending Club's own
  proprietary sub-grade at the same approval rate.

Charts are in `reports/figures/`: ROC, calibration, rank ordering, SHAP importance and the
cutoff strategy curves. Sample reason codes for declined applicants are in
`reports/reason_codes_sample.csv`.

## How to run

1. Download `accepted_2007_to_2018Q4.csv.gz` from the
   [Lending Club dataset on Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
   and put it in `data/raw/`.
2. Install dependencies (Python 3.12):

   ```bash
   python -m venv .venv
   .venv\Scripts\python -m pip install -r requirements.txt
   ```

3. Run everything:

   ```bash
   .venv\Scripts\python -m src.pipeline
   ```

   The first run reads the 1.6 GB file and caches a small table in `data/processed/`. After editing
   `src/config.py`, add `--rebuild`.

To check the code without the real data, generate a fake file and point the pipeline at it:

```bash
.venv\Scripts\python tests\make_fake_data.py data\fake\fake_loans.csv.gz
.venv\Scripts\python -m src.pipeline --raw data\fake\fake_loans.csv.gz --reports data\fake\reports
```

## Project layout

```
src/
  config.py       sample definition, features, monotone directions, profit assumptions
  data_prep.py    raw file -> one row per resolved loan, target, engineered features
  scorecard.py    WoE binning, information value, logistic regression, points table
  gbm.py          LightGBM challenger with monotone constraints
  evaluation.py   AUC / Gini / KS, calibration, decile tables and charts
  explain.py      SHAP values and adverse action reason codes
  strategy.py     cutoff curves and swap-set analysis
  pipeline.py     runs all steps and writes reports/
tests/
  make_fake_data.py  small fake dataset for a quick end-to-end check
```

## Limitations

- **Reject inference.** Lending Club only shows loans it approved, so the model never sees the
  applicants Lending Club declined. Rates on the approved population are optimistic for a lender
  that would approve deeper.
- **Profit is simplified.** Realized cash collected minus principal minus a flat cost of funds;
  no servicing, acquisition or capital costs.
- **No bureau trended data.** Real lenders use months of tradeline history; this dataset has a
  snapshot at application.
- **Lending Club grade is a strong benchmark.** It was built with more data than is public, so the
  goal is not to beat it, but to build an independent, explainable view of risk.
