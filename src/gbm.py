"""LightGBM challenger model with business-sense monotone constraints."""
import lightgbm as lgb
import pandas as pd

from . import config


def learn_categories(train):
    """Category levels seen in training; later samples are encoded with the same levels."""
    return {c: sorted(train[c].dropna().unique()) for c in config.CATEGORICAL_FEATURES}


def to_gbm_frame(df, categories):
    X = df[config.FEATURES].copy()
    for col in config.CATEGORICAL_FEATURES:
        X[col] = pd.Categorical(X[col], categories=categories[col])
    return X


def train_lgbm(X_train, y_train, X_valid, y_valid):
    monotone = [config.MONOTONE.get(c, 0) for c in X_train.columns]
    model = lgb.LGBMClassifier(
        n_estimators=3000,
        learning_rate=0.02,
        num_leaves=31,
        min_child_samples=500,     # large leaves: credit data is noisy, small leaves overfit
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        monotone_constraints=monotone,
        random_state=config.RANDOM_STATE,
        verbose=-1,
    )
    # Stop adding trees once AUC on the in-time holdout stops improving.
    model.fit(X_train, y_train, eval_X=X_valid, eval_y=y_valid, eval_metric="auc",
              callbacks=[lgb.early_stopping(200, verbose=False)])
    return model
