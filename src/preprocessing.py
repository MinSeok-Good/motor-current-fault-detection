import argparse
import csv
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis


def parse_metadata(filename):
    rows = []
    with open(filename, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for _ in range(9):
            rows.append(next(reader))
    return {
        "date": rows[0][1], "filename": rows[1][1], "label": rows[2][1],
        "label_no": rows[3][1], "equipment": rows[4][1],
        "rpm": float(rows[4][2]), "power_kw": float(rows[4][3]),
        "rated_current": float(rows[4][4]), "period": rows[5][1],
        "sample_rate": int(rows[6][1]), "rms_R_raw": float(rows[7][1]),
        "rms_S_raw": float(rows[7][2]), "rms_T_raw": float(rows[7][3]),
        "data_length": int(rows[8][1]),
    }


def load_signal(filename):
    return pd.read_csv(filename, header=None, skiprows=9, usecols=[0,1,2,3], names=["time","R","S","T"])


def extract_signal_features(signal):
    signal = np.asarray(signal, dtype=float)
    rms = np.sqrt(np.mean(signal ** 2))
    std = np.std(signal)
    p2p = np.max(signal) - np.min(signal)
    peak = np.max(np.abs(signal))
    return {
        "rms": rms,
        "std": std,
        "p2p": p2p,
        "skewness": skew(signal),
        "kurtosis": kurtosis(signal),
        "crest_factor": peak / rms if rms != 0 else 0.0,
        "zero_ratio": np.mean(signal == 0),
    }


def make_feature_row(filename):
    meta = parse_metadata(filename)
    signal_df = load_signal(filename)
    row = {k: meta[k] for k in ["date","label","equipment","rpm","power_kw","rated_current","sample_rate","data_length"]}
    row["path"] = str(filename)
    rms_values, p2p_values, zero_ratios = [], [], []
    for phase in ["R","S","T"]:
        feat = extract_signal_features(signal_df[phase])
        for k, v in feat.items(): row[f"{phase}_{k}"] = v
        rms_values.append(feat["rms"]); p2p_values.append(feat["p2p"]); zero_ratios.append(feat["zero_ratio"])
    mean_rms, mean_p2p = np.mean(rms_values), np.mean(p2p_values)
    row["mean_rms"] = mean_rms
    row["mean_p2p"] = mean_p2p
    row["max_zero_ratio"] = max(zero_ratios)
    row["rms_imbalance_ratio"] = (max(rms_values)-min(rms_values))/mean_rms if mean_rms else np.nan
    row["p2p_imbalance_ratio"] = (max(p2p_values)-min(p2p_values))/mean_p2p if mean_p2p else np.nan
    row["low_signal_candidate"] = (mean_rms < 0.5) or (row["max_zero_ratio"] > 0.5)
    return row


def build_dataset(data_dir):
    files = sorted(Path(data_dir).rglob("*.csv"))
    rows = []
    for i, file in enumerate(files, 1):
        try: rows.append(make_feature_row(file))
        except Exception as e: print(f"[WARN] {file}: {e}")
        if i % 100 == 0: print(f"{i}/{len(files)} processed")
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--output", default="ai_ready_features.csv")
    args = p.parse_args()
    df = build_dataset(args.data_dir)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print("Saved:", args.output, "rows:", len(df))

if __name__ == "__main__": main()
