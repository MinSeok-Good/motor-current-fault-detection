"""Run the same code as the verified notebooks from a command line."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, help="Folder containing both feature CSVs")
    parser.add_argument("--stage", choices=["models", "statistics", "all"], default="all")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.data_dir:
        os.environ["MOTOR_FEATURE_DIR"] = str(args.data_dir.resolve())
    os.chdir(root)
    notebooks = []
    if args.stage in ("models", "all"):
        notebooks.append("02_verify_from_clean_csv.ipynb")
    if args.stage in ("statistics", "all"):
        notebooks.append("03_verify_statistics_and_environment.ipynb")
    for name in notebooks:
        notebook = json.loads((root / "notebooks" / name).read_text(encoding="utf-8"))
        namespace = {"__name__": "__main__"}
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                code = compile("".join(cell["source"]), f"{name}:cell{index}", "exec")
                exec(code, namespace)


if __name__ == "__main__":
    main()
