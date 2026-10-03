import argparse
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from xgboost import XGBClassifier

PHASE_7 = ["R_rms","S_rms","T_rms","R_p2p","S_p2p","T_p2p","rms_imbalance_ratio"]


def prepare_dsf(df):
    df = df.copy()
    if "low_signal_candidate" in df.columns:
        df = df[~df["low_signal_candidate"].astype(bool)].copy()
    tokens = df["path"].str.extract(r"_(\d{8})_\d{6}_", expand=False)
    df["day"] = pd.to_datetime(tokens, format="%Y%m%d", errors="raise").dt.date
    if df["day"].isna().any():
        raise ValueError("Cannot extract collection date from path")
    dsf = df[df["equipment"] == "L-DSF-01"].copy()
    overlap_days = []
    for day, g in dsf.groupby("day"):
        if {"정상","축정렬불량"}.issubset(set(g["label"])): overlap_days.append(day)
    dsf = dsf[dsf["day"].isin(overlap_days)].copy()
    dsf["target"] = (dsf["label"] == "축정렬불량").astype(int)
    return dsf


def evaluate_logo(model_df):
    y, groups = model_df["target"], model_df["day"]
    candidates = {
        "R_rms_stump": (["R_rms"], DecisionTreeClassifier(max_depth=1, class_weight="balanced", random_state=42)),
        "Logistic_phase7": (PHASE_7, Pipeline([("scaler", StandardScaler()),("model", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))])),
        "XGBoost_phase7": (PHASE_7, None),
    }
    logo, rows = LeaveOneGroupOut(), []
    for name, (features, base_model) in candidates.items():
        X = model_df[features]
        all_true, all_pred, fold_ba = [], [], []
        for train_idx, test_idx in logo.split(X, y, groups=groups):
            Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
            ytr, yte = y.iloc[train_idx], y.iloc[test_idx]
            if name == "XGBoost_phase7":
                model = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                                      scale_pos_weight=(ytr==0).sum()/(ytr==1).sum(), eval_metric="logloss", random_state=42, n_jobs=-1)
            else: model = base_model
            model.fit(Xtr, ytr); pred = model.predict(Xte)
            all_true.extend(yte.tolist()); all_pred.extend(pred.tolist())
            fold_ba.append(balanced_accuracy_score(yte, pred))
        rows.append({"model":name,"OOF_BA":balanced_accuracy_score(all_true,all_pred),"OOF_F1_macro":f1_score(all_true,all_pred,average="macro"),
                     "Day_BA_mean":np.mean(fold_ba),"Day_BA_std":np.std(fold_ba)})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--features_csv", required=True); args = p.parse_args()
    df = pd.read_csv(args.features_csv, encoding="utf-8-sig"); model_df = prepare_dsf(df)
    result = evaluate_logo(model_df); print(result.to_string(index=False))
    stump = DecisionTreeClassifier(max_depth=1, class_weight="balanced", random_state=42).fit(model_df[["R_rms"]], model_df["target"])
    print("\n", export_text(stump, feature_names=["R_rms"]))

if __name__ == "__main__": main()

