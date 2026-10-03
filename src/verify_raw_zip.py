"""Compare selected raw ZIP members with existing feature and clean CSVs."""
import argparse
from collections import defaultdict
import csv
import io
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis


def basename(path):
    return str(path).replace("\\", "/").rsplit("/", 1)[-1]


def verify(zip_path, before_path, clean_path, out):
    before = pd.read_csv(before_path, encoding="utf-8-sig")
    clean = pd.read_csv(clean_path, encoding="utf-8-sig")
    if before.path.duplicated().any() or clean.path.duplicated().any():
        raise ValueError("Duplicate paths in feature CSV")
    clean_paths = set(clean.path)
    if not clean_paths.issubset(set(before.path)):
        raise ValueError("Clean CSV contains unknown paths")
    reports = []
    with ZipFile(zip_path) as archive:
        members = defaultdict(list)
        for info in archive.infolist():
            if not info.is_dir() and info.filename.lower().endswith(".csv"):
                members[basename(info.filename)].append(info)
        for number, (_, expected) in enumerate(before.iterrows(), 1):
            name = basename(expected.path)
            report = {"filename": name, "path": expected.path}
            try:
                matches = members.get(name, [])
                if len(matches) != 1:
                    raise ValueError(f"Expected one ZIP member, found {len(matches)}")
                text = archive.read(matches[0]).decode("utf-8-sig")
                meta = list(csv.reader(io.StringIO(text)))[:9]
                signal = pd.read_csv(io.StringIO(text), header=None, skiprows=9,
                                     usecols=[0, 1, 2, 3], names=["time", "R", "S", "T"])
                if len(signal) != int(meta[8][1]) or not np.isfinite(signal.to_numpy(dtype=float)).all():
                    raise ValueError("Invalid signal length or nonfinite values")
                actual = {"equipment": meta[4][1], "rpm": float(meta[4][2]),
                          "power_kw": float(meta[4][3]), "rated_current": float(meta[4][4]),
                          "label": meta[2][1]}
                rms_values, zeros = [], []
                for phase in ["R", "S", "T"]:
                    values = signal[phase].to_numpy(dtype=float)
                    rms = float(np.sqrt(np.mean(values ** 2)))
                    rms_values.append(rms)
                    zeros.append(float(np.mean(values == 0)))
                    features = {"rms": rms, "std": float(np.std(values)),
                                "p2p": float(np.ptp(values)), "skewness": float(skew(values)),
                                "kurtosis": float(kurtosis(values)),
                                "crest_factor": float(np.max(np.abs(values)) / rms) if rms else 0}
                    actual.update({f"{phase}_{key}": value for key, value in features.items()})
                mean_rms = float(np.mean(rms_values))
                actual["rms_imbalance_ratio"] = (max(rms_values) - min(rms_values)) / mean_rms if mean_rms else 0
                differences, mismatches = [], []
                for key, value in actual.items():
                    if isinstance(value, str):
                        same = value == expected[key]
                    else:
                        differences.append(abs(value - float(expected[key])))
                        same = np.isclose(value, float(expected[key]), rtol=0, atol=1e-8)
                    if not same:
                        mismatches.append(key)
                should_remove = mean_rms < 0.5 or max(zeros) > 0.5
                removed = expected.path not in clean_paths
                report.update(status="ok", feature_match=not mismatches,
                              mismatch_columns=",".join(mismatches), max_abs_difference=max(differences),
                              mean_rms=mean_rms, max_zero_ratio=max(zeros), should_remove=should_remove,
                              actually_removed=removed, filter_match=should_remove == removed)
            except Exception as error:
                report.update(status="error", error=f"{type(error).__name__}: {error}")
            reports.append(report)
            if number % 100 == 0 or number == len(before):
                print(f"{number:,} / {len(before):,}", flush=True)
    result = pd.DataFrame(reports)
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / "raw_zip_verification.csv", index=False, encoding="utf-8-sig")
    ok = result.loc[result.status.eq("ok")]
    retained = before.loc[before.path.isin(clean_paths)].sort_values("path").reset_index(drop=True)
    unchanged = retained.equals(clean.sort_values("path").reset_index(drop=True))
    summary = {"target_files": len(before), "read_files": len(ok), "errors": len(result) - len(ok),
               "feature_mismatches": int((~ok.feature_match.astype(bool)).sum()) if len(ok) else 0,
               "filter_mismatches": int((~ok.filter_match.astype(bool)).sum()) if len(ok) else 0,
               "clean_values_unchanged": unchanged,
               "expected_removed": int(ok.should_remove.sum()) if len(ok) else 0,
               "actual_removed": len(before) - len(clean)}
    print(summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/reproduction"))
    args = parser.parse_args()
    summary = verify(args.zip, args.data_dir / "ai_ready_time_features.csv",
                     args.data_dir / "ai_ready_time_features_clean.csv", args.output_dir)
    if summary["errors"] or summary["feature_mismatches"] or summary["filter_mismatches"] or not summary["clean_values_unchanged"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
