"""Build the LaTeX tables of the phase 1 report from the notebook outputs.

Reads the CSV and JSON files that notebooks/01_phase1_training.ipynb writes to
notebooks/outputs/phase1/ and that notebooks/02_phase2_tuning.ipynb writes to
notebooks/outputs/phase2/, and writes one tabular per table to docs/tables/, so
the numbers of the report are never typed by hand. Standard library only:

    python3 docs/build_tables.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "notebooks" / "outputs" / "phase1"
PHASE2 = ROOT / "notebooks" / "outputs" / "phase2"
TABLES = ROOT / "docs" / "tables"

MODELS = {
    "logistic_regression": "Logistic Regression",
    "naive_bayes": "Naive Bayes",
    "decision_tree": "Decision Tree",
}


def read_csv(name: str, folder: Path = OUTPUTS) -> list[dict[str, str]]:
    with open(folder / name, newline="", encoding="utf-8") as file:
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


def confusion_matrix(key: str, folder: Path = OUTPUTS) -> str:
    records = read_csv(f"confusion_matrix_{key}.csv", folder)
    classes = [column for column in records[0] if column != "reference"]
    rows = [[r["reference"], *(count(r[c]) for c in classes), count(sum(int(r[c]) for c in classes))] for r in records]
    total = ["Total", *(count(sum(int(r[c]) for r in records)) for c in classes),
             count(sum(int(r[c]) for r in records for c in classes))]
    return tabular("c" + "r" * (len(classes) + 1), ["Ref.", *classes, "Total"], rows, [total])


# Phase 2: hyper-parameter optimisation ------------------------------------------------

TUNED_MODELS = {
    **MODELS,
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
}


def latex(text: str) -> str:
    return str(text).replace("_", "\\_").replace("%", "\\%")


def param_value(value: str) -> str:
    """A grid value as printed in cv_results (None, True, 0.5, 'balanced')."""
    value = value.strip()
    try:
        number = float(value)
    except ValueError:
        return latex(value)
    if number.is_integer() and "e" not in value.lower():
        return str(int(number))
    return f"{number:.0e}" if abs(number) < 1e-3 else f"{number:g}"


def grid_values(values: list[str]) -> str:
    numbers = []
    for value in values:
        try:
            numbers.append(float(value))
        except ValueError:
            break
    if len(numbers) == len(values) and len(values) > 5:
        return f"{len(values)} values, {param_value(str(min(numbers)))} to {param_value(str(max(numbers)))} (log scale)"
    return ", ".join(param_value(v) for v in values)


def search_space() -> str:
    rows = []
    for key, name in TUNED_MODELS.items():
        records = read_csv(f"cv_results_{key}.csv", PHASE2)
        params = [column for column in records[0] if column.startswith("param_")]
        lines = []
        for column in params:
            seen = list(dict.fromkeys(r[column] or "None" for r in records))
            lines.append(f"\\texttt{{{latex(column.removeprefix('param_').split('__')[-1])}}}: {grid_values(seen)}")
        rows.append([name, "\\makecell[l]{" + " \\\\ ".join(lines) + "}", str(len(records))])
    rows = [row[:-1] + [row[-1] + " \\\\ \\addlinespace"] for row in rows[:-1]] + rows[-1:]
    table = tabular("llr", ["Model", "Candidate values", "Combinations"], rows)
    return table.replace("\\addlinespace \\\\", "\\addlinespace")


def selected_params() -> str:
    tuned = json.loads((PHASE2 / "best_params.json").read_text(encoding="utf-8"))
    rows = []
    for key, name in TUNED_MODELS.items():
        params = tuned[key]["params"]
        text = ", ".join(f"\\texttt{{{latex(k.split('__')[-1])}}} = {param_value(str(v))}" for k, v in params.items())
        rows.append([name, text])
    return tabular(">{\\raggedright\\arraybackslash}l>{\\raggedright\\arraybackslash}p{11cm}", ["Model", "Selected configuration"], rows)


def fold_shares() -> str:
    names = {int(r["Class"]): r["name"] for r in read_csv("class_distribution.csv")}
    rows = [
        [r["Class"], names[int(r["Class"])],
         *(f"{float(r[c]) * 100:.2f}" for c in ("training share", "min fold share", "max fold share"))]
        for r in read_csv("fold_class_shares.csv", PHASE2)
    ]
    return tabular("clrrr", ["Class", "Land cover", "Training (\\%)", "Min. fold (\\%)", "Max. fold (\\%)"], rows)


def short_name(model: str) -> str:
    return model.split(" (")[0]


def tuned_cv() -> str:
    rows = []
    for r in read_csv("summary.csv", PHASE2):
        row = [short_name(r["model"]),
               f"{float(r['cv_f1_macro_mean']):.4f} $\\pm$ {float(r['cv_f1_macro_std']):.4f}",
               f"{float(r['cv_accuracy_mean']):.4f}", f"{float(r['cv_kappa_mean']):.4f}"]
        if r["selected"] == "True":
            row = [f"\\textbf{{{cell}}}" for cell in row]
        rows.append(row)
    return tabular("lrrr", ["Model", "Macro F1 (mean $\\pm$ sd)", "Overall accuracy", "Kappa"], rows)


def tuned_holdout() -> str:
    # Macro recall over the evaluated classes equals the balanced accuracy, so it is shown once.
    fields = ["test_overall_accuracy", "test_kappa", "test_precision_macro",
              "test_recall_macro", "test_f1_macro", "test_f1_weighted"]
    rows = []
    for r in read_csv("summary.csv", PHASE2):
        row = [short_name(r["model"]), *(f"{float(r[f]):.4f}" for f in fields)]
        if r["selected"] == "True":
            row = [f"\\textbf{{{cell}}}" for cell in row]
        rows.append(row)
    header = ["Model", "OA", "Kappa", "Macro precision", "Macro recall", "Macro F1", "Weighted F1"]
    return tabular("lrrrrrr", header, rows)


def phase_comparison() -> str:
    pairs = [("OA", "phase 1 OA", "phase 2 OA"), ("Kappa", "phase 1 kappa", "phase 2 kappa"),
             ("Macro F1", "phase 1 macro-F1", "phase 2 macro-F1")]
    rows = []
    for r in read_csv("phase1_vs_phase2.csv", PHASE2):
        row = [r["model"]]
        for _, before, after in pairs:
            delta = float(r[after]) - float(r[before])
            row += [f"{float(r[before]):.4f}", f"{float(r[after]):.4f}", f"{delta:+.4f}"]
        rows.append(row)
    groups = " & ".join(f"\\multicolumn{{3}}{{c}}{{{label}}}" for label, _, _ in pairs)
    rules = " ".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(pairs)))
    stages = ["Default", "Tuned", "$\\Delta$"] * len(pairs)
    table = tabular("l" + "r" * 9, ["", *stages], rows)
    return table.replace("\\toprule\n", f"\\toprule\n & {groups} \\\\\n{rules}\n", 1)


def f1_per_class() -> str:
    records = read_csv("f1_per_class.csv", PHASE2)
    models = [c for c in records[0] if c not in ("Class", "name") and not c.startswith("Most frequent")]
    rows = [[r["Class"], r["name"], *(f"{float(r[m]):.3f}" for m in models)] for r in records]
    header = ["\\makecell{" + short_name(m).replace(" ", " \\\\ ") + "}" for m in models]
    return tabular("cl" + "r" * len(models), ["Class", "Land cover", *header], rows)


def phase2_tables() -> dict[str, str]:
    selected = json.loads((PHASE2 / "metrics.json").read_text(encoding="utf-8"))["selected"]
    return {
        "p2_search_space": search_space(), "p2_selected_params": selected_params(),
        "p2_fold_shares": fold_shares(), "p2_cv": tuned_cv(), "p2_holdout": tuned_holdout(),
        "p2_phase_comparison": phase_comparison(), "p2_f1_per_class": f1_per_class(),
        "p2_confusion_matrix_selected": confusion_matrix(selected, PHASE2),
    }


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    tables = {"class_distribution": class_distribution(), "split": split(), "summary": summary(), "recall": recall()}
    tables.update({f"confusion_matrix_{key}": confusion_matrix(key) for key in MODELS})
    if (PHASE2 / "metrics.json").exists():
        tables.update(phase2_tables())
    for name, content in tables.items():
        (TABLES / f"{name}.tex").write_text(content, encoding="utf-8")
        print(f"wrote docs/tables/{name}.tex")


if __name__ == "__main__":
    main()
