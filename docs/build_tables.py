"""Build the LaTeX tables of the phase 1 report from the notebook outputs.

Reads the CSV and JSON files that notebooks/01_phase1_training.ipynb writes to
notebooks/outputs/phase1/ and writes one tabular per table to docs/tables/, so
the numbers of the report are never typed by hand. Standard library only:

    python3 docs/build_tables.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "notebooks" / "outputs" / "phase1"
TABLES = ROOT / "docs" / "tables"

MODELS = {
    "logistic_regression": "Logistic Regression",
    "naive_bayes": "Naive Bayes",
    "decision_tree": "Decision Tree",
}


def read_csv(name: str) -> list[dict[str, str]]:
    with open(OUTPUTS / name, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def count(value: str) -> str:
    """Integer with thousands separators."""
    return f"{int(float(value)):,}"


def tabular(columns: str, header: list[str], rows: list[list[str]], footer: list[list[str]] = ()) -> str:
    lines = [f"\\begin{{tabular}}{{{columns}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    lines += [" & ".join(row) + " \\\\" for row in rows]
    if footer:
        lines.append("\\midrule")
        lines += [" & ".join(row) + " \\\\" for row in footer]
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def class_distribution() -> str:
    records = read_csv("class_distribution.csv")
    rows = [
        [r["Class"], r["name"], count(r["polygons"]), count(r["samples"]), f"{float(r['share']) * 100:.2f}"]
        for r in records
    ]
    total = ["", "Total", count(sum(int(r["polygons"]) for r in records)),
             count(sum(int(r["samples"]) for r in records)), "100.00"]
    return tabular("clrrr", ["Class", "Land cover", "Polygons", "Pixels", "Share (\\%)"], rows, [total])


def split() -> str:
    records = read_csv("split.csv")
    fields = ["train_polygons", "train_samples", "test_polygons", "test_samples"]
    rows = [[r["Class"], *(count(r[field]) for field in fields)] for r in records]
    total = ["Total", *(count(sum(int(r[field]) for r in records)) for field in fields)]
    header = ["Class", "Train polygons", "Train pixels", "Test polygons", "Test pixels"]
    return tabular("crrrr", header, rows, [total])


def summary() -> str:
    metrics = json.loads((OUTPUTS / "metrics.json").read_text(encoding="utf-8"))
    rows = [
        [r["model"], f"{float(r['overall_accuracy']):.4f}", f"{float(r['kappa']):.4f}", f"{float(r['f1_macro']):.4f}"]
        for r in read_csv("summary.csv")
    ]
    baseline = [f"Always class {metrics['majority_class']} (baseline)",
                f"{metrics['majority_baseline_accuracy']:.4f}", "0.0000", "--"]
    return tabular("lrrr", ["Model", "Overall accuracy", "Kappa", "Macro F1"], rows, [baseline])


def recall() -> str:
    names = list(MODELS.values())
    rows = [
        [r["Class"], r["name"], count(r["test_samples"]),
         *(f"{float(r[name]):.3f}" if r[name] else "--" for name in names)]
        for r in read_csv("recall_per_class.csv")
    ]
    return tabular("clrrrr", ["Class", "Land cover", "Test pixels", *names], rows)


def confusion_matrix(key: str) -> str:
    records = read_csv(f"confusion_matrix_{key}.csv")
    classes = [column for column in records[0] if column != "reference"]
    rows = [[r["reference"], *(count(r[c]) for c in classes), count(sum(int(r[c]) for c in classes))] for r in records]
    total = ["Total", *(count(sum(int(r[c]) for r in records)) for c in classes),
             count(sum(int(r[c]) for r in records for c in classes))]
    return tabular("c" + "r" * (len(classes) + 1), ["Ref.", *classes, "Total"], rows, [total])


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    tables = {"class_distribution": class_distribution(), "split": split(), "summary": summary(), "recall": recall()}
    tables.update({f"confusion_matrix_{key}": confusion_matrix(key) for key in MODELS})
    for name, content in tables.items():
        (TABLES / f"{name}.tex").write_text(content, encoding="utf-8")
        print(f"wrote docs/tables/{name}.tex")


if __name__ == "__main__":
    main()
