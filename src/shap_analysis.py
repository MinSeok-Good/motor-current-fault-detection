import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBClassifier
from modeling import PHASE_7, prepare_dsf


def main():
    p = argparse.ArgumentParser(); p.add_argument("--features_csv", required=True); p.add_argument("--output_dir", default="figures"); args = p.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.features_csv, encoding="utf-8-sig"); mdf = prepare_dsf(df)
    X, y = mdf[PHASE_7], mdf["target"]
    model = XGBClassifier(n_estimators=300,max_depth=4,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,
                          scale_pos_weight=(y==0).sum()/(y==1).sum(),eval_metric="logloss",random_state=42,n_jobs=-1).fit(X,y)
    dm = xgb.DMatrix(X, feature_names=X.columns.tolist())
    contrib = model.get_booster().predict(dm, pred_contribs=True)
    shap_df = pd.DataFrame(contrib[:,:-1], columns=X.columns)
    imp = shap_df.abs().mean().sort_values(ascending=False)
    imp.to_csv(out/"shap_importance.csv", encoding="utf-8-sig")
    plt.figure(figsize=(8,5)); imp.sort_values().plot(kind="barh"); plt.xlabel("Mean |SHAP value|"); plt.ylabel("Feature"); plt.title("XGBoost SHAP Feature Importance"); plt.tight_layout(); plt.savefig(out/"shap_feature_importance.png", dpi=150); plt.close()
    plt.figure(figsize=(8,5)); plt.scatter(X["R_rms"], shap_df["R_rms"], alpha=0.6); plt.axhline(0,ls="--"); plt.axvline(2.30,ls="--"); plt.xlabel("R_rms"); plt.ylabel("SHAP contribution"); plt.title("R_rms SHAP Contribution"); plt.tight_layout(); plt.savefig(out/"r_rms_shap_contribution.png", dpi=150); plt.close()
    print(imp)

if __name__ == "__main__": main()
