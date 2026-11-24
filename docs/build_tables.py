"""Build the LaTeX tables of the phase 1 and final reports from the notebook outputs.

Reads the CSV and JSON files that the notebooks write to notebooks/outputs/
(phase1/, phase2/ and inference/ for the full-scene map) and
writes one tabular per table to docs/tables/, so the numbers of the reports are
never typed by hand. The final_* tables feed docs/final_report.tex. Standard
library only:

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


def phase1_report_rows() -> list[dict]:
    """Summary rows of the models in the phase 1 report (the five tuned there and the baseline).

    The phase 2 notebook later gained KNN, the SVM and the neural network; they belong to the
    final report only, so the delivered phase 1 tables stay as they were. The best of these
    models by cross-validated macro-F1 is flagged as selected."""
    names = {short_name(n) for n in TUNED_MODELS.values()} | {"Most frequent"}
    rows = [r for r in read_csv("summary.csv", PHASE2) if short_name(r["model"]) in names]
    best = max((r for r in rows if not r["model"].startswith("Most frequent")), key=lambda r: float(r["cv_f1_macro_mean"]))
    return [{**r, "selected": str(r is best)} for r in rows]


def phase1_report_selected() -> str:
    best = next(r for r in phase1_report_rows() if r["selected"] == "True")
    return next(k for k, n in TUNED_MODELS.items() if short_name(n) == short_name(best["model"]))


def tuned_cv() -> str:
    rows = []
    for r in phase1_report_rows():
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
    for r in phase1_report_rows():
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
    names = {short_name(n) for n in TUNED_MODELS.values()}
    models = [c for c in records[0] if short_name(c) in names]
    rows = [[r["Class"], r["name"], *(f"{float(r[m]):.3f}" for m in models)] for r in records]
    header = ["\\makecell{" + short_name(m).replace(" ", " \\\\ ") + "}" for m in models]
    return tabular("cl" + "r" * len(models), ["Class", "Land cover", *header], rows)


def phase2_tables() -> dict[str, str]:
    selected = phase1_report_selected()
    return {
        "p2_search_space": search_space(), "p2_selected_params": selected_params(),
        "p2_fold_shares": fold_shares(), "p2_cv": tuned_cv(), "p2_holdout": tuned_holdout(),
        "p2_phase_comparison": phase_comparison(), "p2_f1_per_class": f1_per_class(),
        "p2_confusion_matrix_selected": confusion_matrix(selected, PHASE2),
    }


# Final report: every architecture of notebook 02, the full-scene inference and the areas -------

INFERENCE = ROOT / "notebooks" / "outputs" / "inference"

# key: (name, abbreviation, folder of its outputs, file key of its confusion matrix)
FINAL_MODELS = {
    "logistic_regression": ("Logistic Regression", "LR", PHASE2, "logistic_regression"),
    "knn": ("K-Nearest Neighbors", "KNN", PHASE2, "knn"),
    "naive_bayes": ("Naive Bayes", "NB", PHASE2, "naive_bayes"),
    "decision_tree": ("Decision Tree", "DT", PHASE2, "decision_tree"),
    "random_forest": ("Random Forest", "RF", PHASE2, "random_forest"),
    "gradient_boosting": ("Gradient Boosting", "GB", PHASE2, "gradient_boosting"),
    "svm": ("Support Vector Machine", "SVM", PHASE2, "svm"),
    "neural_network": ("Neural Network (MLP)", "ANN", PHASE2, "neural_network"),
}
BASELINE = ("dummy", "Always class 5 (baseline)")


def matrix_metrics(folder: Path, key: str) -> dict:
    """Holdout metrics of one model computed from its saved confusion matrix.

    Every model goes through the same formulas, so the comparison does not depend
    on how each notebook computed its summary. Averages are over the classes with
    reference pixels in the test set (2 to 9)."""
    records = read_csv(f"confusion_matrix_{key}.csv", folder)
    labels = [c for c in records[0] if c != "reference"]
    m = [[int(r[c]) for c in labels] for r in records]
    n = sum(map(sum, m))
    rows = [sum(row) for row in m]
    cols = [sum(m[i][j] for i in range(len(m))) for j in range(len(labels))]
    diag = [m[i][i] for i in range(len(labels))]
    evaluated = [i for i in range(len(labels)) if rows[i] > 0]
    per_class = {}
    for i in evaluated:
        precision = diag[i] / cols[i] if cols[i] else 0.0
        recall = diag[i] / rows[i]
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[labels[i]] = {"precision": precision, "recall": recall, "f1": f1, "support": rows[i]}
    support = sum(v["support"] for v in per_class.values())
    observed = sum(diag) / n
    expected = sum(rows[i] * cols[i] for i in range(len(labels))) / n ** 2
    macro = {k: sum(v[k] for v in per_class.values()) / len(per_class) for k in ("precision", "recall", "f1")}
    return {
        "overall_accuracy": observed, "kappa": (observed - expected) / (1 - expected),
        "precision_macro": macro["precision"], "recall_macro": macro["recall"], "f1_macro": macro["f1"],
        "f1_weighted": sum(v["f1"] * v["support"] for v in per_class.values()) / support,
        "per_class": per_class,
    }


def check_against_summaries(metrics: dict) -> None:
    """The matrix-derived values must reproduce what the notebooks reported."""
    by_model = {r["model"]: r for r in read_csv("summary.csv", PHASE2)}
    pairs = [("overall_accuracy", "test_overall_accuracy"), ("kappa", "test_kappa"),
             ("f1_macro", "test_f1_macro"), ("f1_weighted", "test_f1_weighted")]
    for key, (name, *_rest) in FINAL_MODELS.items():
        row = next(r for model, r in by_model.items() if model.startswith(name.split(" (")[0]))
        for mine, theirs in pairs:
            assert abs(metrics[key][mine] - float(row[theirs])) < 1e-9, (key, mine, metrics[key][mine], row[theirs])


def best_cv_row(key: str) -> dict:
    records = read_csv(f"cv_results_{key}.csv", PHASE2)
    return max(records, key=lambda r: float(r["mean_test_f1_macro"]))


def phase2_selected() -> str:
    return json.loads((PHASE2 / "metrics.json").read_text(encoding="utf-8"))["selected"]


def bold_if(row: list[str], condition: bool) -> list[str]:
    return [f"\\textbf{{{cell}}}" for cell in row] if condition else row


def final_search_space() -> str:
    rows = []
    for key, (name, *_rest) in FINAL_MODELS.items():
        records = read_csv(f"cv_results_{key}.csv", PHASE2)
        params = [c for c in records[0] if c.startswith("param_")]
        label = lambda column: column.removeprefix("param_").split("__")[-1]  # noqa: E731
        lines = [f"\\texttt{{{latex(label(c))}}}: {grid_values(list(dict.fromkeys(r[c] or 'None' for r in records)))}"
                 for c in params]
        rows.append([name, "\\makecell[l]{" + " \\\\ ".join(lines) + "}", str(len(records))])
    rows = [row[:-1] + [row[-1] + " \\\\ \\addlinespace"] for row in rows[:-1]] + rows[-1:]
    table = tabular("llr", ["Model", "Candidate values", "Combinations"], rows)
    return table.replace("\\addlinespace \\\\", "\\addlinespace")


def final_selected_params() -> str:
    tuned = json.loads((PHASE2 / "best_params.json").read_text(encoding="utf-8"))
    rows = []
    for key, (name, *_rest) in FINAL_MODELS.items():
        params = tuned[key]["params"]
        # JSON stores tuples (hidden layer sizes) as lists; show them as in the search space.
        text = ", ".join(f"\\texttt{{{latex(k.split('__')[-1])}}} = {param_value(str(tuple(v) if isinstance(v, list) else v))}"
                         for k, v in params.items())
        rows.append([name, text])
    return tabular(">{\\raggedright\\arraybackslash}l>{\\raggedright\\arraybackslash}p{10.5cm}",
                   ["Model", "Selected configuration"], rows)


def final_cv() -> str:
    """Validation scores of the selected configurations over the ten validations."""
    selected = phase2_selected()
    rows = []
    for key, (name, *_rest) in FINAL_MODELS.items():
        best = best_cv_row(key)
        row = [name, f"{float(best['mean_test_f1_macro']):.4f} $\\pm$ {float(best['std_test_f1_macro']):.4f}",
               f"{float(best['mean_test_accuracy']):.4f}", f"{float(best['mean_test_kappa']):.4f}"]
        rows.append(bold_if(row, key == selected))
    header = ["Model", "Macro F1", "OA", "Kappa"]
    return tabular("lrrr", header, rows)


def final_holdout(metrics: dict) -> str:
    """The summary table required by the brief: every architecture on the same test set."""
    selected = phase2_selected()
    fields = ["overall_accuracy", "kappa", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"]
    order = sorted(FINAL_MODELS, key=lambda k: -metrics[k]["f1_macro"])
    rows = [bold_if([FINAL_MODELS[k][0], *(f"{metrics[k][f]:.4f}" for f in fields)], k == selected) for k in order]
    base = matrix_metrics(PHASE2, BASELINE[0])
    footer = [[BASELINE[1], *(f"{base[f]:.4f}" for f in fields)]]
    header = ["Model", "OA", "Kappa", "\\makecell{Macro\\\\precision}", "\\makecell{Macro\\\\recall}",
              "\\makecell{Macro\\\\F1}", "\\makecell{Weighted\\\\F1}"]
    return tabular("lrrrrrr", header, rows, footer)


def final_metric_per_class(metrics: dict, metric: str) -> str:
    """One per-class metric (precision, recall or f1) of every model; the best value of each class in bold."""
    names = {int(r["Class"]): r["name"] for r in read_csv("class_distribution.csv")}
    classes = list(metrics["logistic_regression"]["per_class"])
    rows = []
    for c in classes:
        values = {k: metrics[k]["per_class"][c][metric] for k in FINAL_MODELS}
        best = max(values.values())
        rows.append([c, names[int(c)], *(f"\\textbf{{{v:.3f}}}" if v == best else f"{v:.3f}" for v in values.values())])
    header = ["Class", "Land cover", *(abbr for _, abbr, _, _ in FINAL_MODELS.values())]
    return tabular("cl" + "r" * len(FINAL_MODELS), header, rows)


def final_class_areas() -> str:
    records = read_csv("class_areas.csv", INFERENCE)
    summary = json.loads((INFERENCE / "summary.json").read_text(encoding="utf-8"))
    rows = [[r["class_id"], r["name"], count(r["pixels"]), f"{float(r['area_ha']):,.2f}",
             f"{float(r['area_km2']):,.2f}", f"{float(r['share_pct']):.2f}"] for r in records]
    pixels = sum(int(r["pixels"]) for r in records)
    footer = [["", "Mapped area", count(pixels), f"{pixels * 0.04:,.2f}", f"{pixels * 0.0004:,.2f}", "100.00"],
              ["", "No data (code 255)", count(summary["nodata_pixels"]), f"{summary['nodata_pixels'] * 0.04:,.2f}",
               f"{summary['nodata_pixels'] * 0.0004:,.2f}", "--"]]
    header = ["Class", "Land cover", "Pixels", "Area (ha)", "Area (km$^2$)", "Share (\\%)"]
    return tabular("clrrrr", header, rows, footer)


def final_tables() -> dict[str, str]:
    metrics = {key: matrix_metrics(folder, file_key) for key, (_, _, folder, file_key) in FINAL_MODELS.items()}
    check_against_summaries(metrics)
    tables = {
        "final_search_space": final_search_space(), "final_selected_params": final_selected_params(),
        "final_cv": final_cv(), "final_holdout": final_holdout(metrics),
        "final_f1_per_class": final_metric_per_class(metrics, "f1"),
        "final_precision_per_class": final_metric_per_class(metrics, "precision"),
        "final_recall_per_class": final_metric_per_class(metrics, "recall"),
        "final_confusion_matrix_selected": confusion_matrix(phase2_selected(), PHASE2),
    }
    if (INFERENCE / "class_areas.csv").exists():
        tables["final_class_areas"] = final_class_areas()
    return tables


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    tables = {"class_distribution": class_distribution(), "split": split(), "summary": summary(), "recall": recall()}
    tables.update({f"confusion_matrix_{key}": confusion_matrix(key) for key in MODELS})
    if (PHASE2 / "metrics.json").exists():
        tables.update(phase2_tables())
    if all((PHASE2 / f"cv_results_{key}.csv").exists() for key in FINAL_MODELS):
        tables.update(final_tables())
    for name, content in tables.items():
        (TABLES / f"{name}.tex").write_text(content, encoding="utf-8")
        print(f"wrote docs/tables/{name}.tex")


if __name__ == "__main__":
    main()
