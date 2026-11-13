"""
Collect every run under results/ into one Excel workbook.

    python scripts/collect_results.py                       # tuning sheet (validation only)
    python scripts/collect_results.py --final-tags final    # also fill the test sheet for these tags

Layout: results/<tag folder>/<run folder>/results.json, with the Part A settings read from the graph files in
results/<tag folder>/graphs*/G_h*.json. The sheets "Tuning (validation)", "Final test", "Comparison" and "Rules" are
rewritten on every call; any other sheet in the workbook is kept.

Settings are chosen on validation error only. Test metrics are written to their own sheet and only for the tags passed
with --final-tags, so a test number never sits next to the numbers a choice is based on.
"""

import argparse
import json
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

HORIZONS = ("3", "6", "12")
MINUTES = {"1": 5, "3": 15, "6": 30, "12": 60}
ALL_HORIZONS = ("1", "3", "6", "12")      # "1" (5 min) exists only for runs trained with horizon 1
OWN_SHEETS = ("Tuning (validation)", "Final test", "Comparison", "Rules")

HEAD_FILL = PatternFill("solid", fgColor="DDE6F0")
BEST_FILL = PatternFill("solid", fgColor="D5ECD4")
BOLD = Font(bold=True)
VS_BASE = "Val MAE vs base (same config, seed)"


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def graph_settings(tag_dir):
    """Part A settings and edge counts of the graphs in a tag folder (empty when the folder has none)."""
    for folder in sorted(tag_dir.glob("graphs*")):
        files = {h: folder / f"G_h{h}.json" for h in HORIZONS}
        if all(p.exists() for p in files.values()):
            info = {h: read_json(p) for h, p in files.items()}
            return info["3"].get("graph_config", {}), {h: info[h].get("edges") for h in HORIZONS}
    return {}, {}


def signature(run):
    """What makes two runs the same run: setting, seed, graph and validation result."""
    v = run["validation"]["mean"]
    return (run["config"], run["seed"], run["hidden"], run["dropout"], run["params"],
            tuple(sorted(run["edges"].items())), round(v["mae"], 6), round(v["rmse"], 6))


def collect(results_dir, include_smoke):
    """All runs under results_dir at any depth. Exact duplicates (the same run pasted under several folder names) are
    reported and dropped, keeping the folder whose tag mentions the run's hidden size, else the shortest path."""
    runs, graph_cfgs = [], {}
    for path in sorted(Path(results_dir).glob("**/results.json")):
        tag_dir = path.parents[1]
        tag = tag_dir.name[len("results_"):] if tag_dir.name.startswith("results_") else tag_dir.name
        if tag == "smoke" and not include_smoke:
            continue
        result = read_json(path)
        config_file = path.parent / "config.json"
        config = read_json(config_file) if config_file.exists() else {}
        train = config.get("train", {})
        model = train.get("model", {})
        graph_cfg, edges = graph_settings(tag_dir)
        if not edges:
            edges = {h: v.get("edges") for h, v in config.get("graphs", {}).items()}
        if graph_cfg:
            graph_cfgs[tuple(sorted(edges.items()))] = graph_cfg
        uses_graph = model.get("use_graph", True)
        runs.append(dict(
            tag=tag, folder=str(path.parent), config=train.get("graph_config", result["run"].split("__")[0]), seed=train.get("seed"),
            graph_cfg=graph_cfg if uses_graph else {}, edges=edges if uses_graph else {}, inferred=False,
            hidden=model.get("hidden"), dropout=model.get("dropout"), params=result.get("parameters"),
            best_epoch=result.get("best_epoch"), epochs=result.get("epochs"),
            minutes=result.get("training_seconds", 0) / 60,
            validation=result["validation"], test=result["test"], gate=result.get("test_gate", {}),
        ))
    for r in runs:
        if r["edges"] and not r["graph_cfg"]:
            match = graph_cfgs.get(tuple(sorted(r["edges"].items())))
            if match:
                r["graph_cfg"], r["inferred"] = match, True

    groups = {}
    for r in runs:
        groups.setdefault(signature(r), []).append(r)
    kept = []
    for same in groups.values():
        same.sort(key=lambda r: (str(r["hidden"]) not in r["tag"], len(r["folder"])))
        kept.append(same[0])
        for dup in same[1:]:
            print(f"duplicate dropped: {dup['folder']}  (same run as {same[0]['folder']})")
    return kept


def style_header(ws, ncols):
    for c in range(1, ncols + 1):
        cell = ws.cell(row=1, column=c)
        cell.font, cell.fill = BOLD, HEAD_FILL
        cell.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")
    ws.row_dimensions[1].height = 48


def set_widths(ws, widths):
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w


def tuning_sheet(wb, runs):
    ws = wb.create_sheet("Tuning (validation)")
    heads = ["Tag", "Config", "Seed",
             "FDR q", "Group corr", "VAR ridge", "Max sources", "VAR lag", "Hidden", "Dropout",
             "Edges h=3", "Edges h=6", "Edges h=12",
             "Val MAE mean", "Val MAE 5 min", "Val MAE 15 min", "Val MAE 30 min", "Val MAE 60 min",
             "Val RMSE mean", "Val RMSE 5 min", "Val RMSE 15 min", "Val RMSE 30 min", "Val RMSE 60 min",
             VS_BASE, "Best epoch", "Epochs run", "Minutes", "Parameters", "Notes"]
    ws.append(heads)
    style_header(ws, len(heads))
    ws.freeze_panes = "D2"

    runs = sorted(runs, key=lambda r: (r["tag"] != "base", r["tag"], r["config"], r["seed"] or 0))
    base_row = {(r["config"], r["seed"]): i for i, r in enumerate(runs, 2) if r["tag"] == "base"}
    mae_col = heads.index("Val MAE mean") + 1
    mae_letter = get_column_letter(mae_col)
    for i, r in enumerate(runs, 2):
        g, v = r["graph_cfg"], r["validation"]
        ws.append([r["tag"], r["config"], r["seed"],
                   g.get("fdr"), g.get("group_corr"), g.get("var_ridge"), g.get("max_sources"), g.get("var_lag"),
                   r["hidden"], r["dropout"],
                   r["edges"].get("3"), r["edges"].get("6"), r["edges"].get("12"),
                   v["mean"]["mae"], v.get("1", {}).get("mae"), v["3"]["mae"], v["6"]["mae"], v["12"]["mae"],
                   v["mean"]["rmse"], v.get("1", {}).get("rmse"), v["3"]["rmse"], v["6"]["rmse"], v["12"]["rmse"],
                   None, r["best_epoch"], r["epochs"], round(r["minutes"], 1), r["params"],
                   "Part A settings matched by edge counts (graphs built in another folder)" if r["inferred"] else None])
        ref = base_row.get((r["config"], r["seed"]))
        if ref and ref != i:
            ws.cell(row=i, column=heads.index(VS_BASE) + 1, value=f"={mae_letter}{i}/{mae_letter}{ref}-1")
    for row in ws.iter_rows(min_row=2, max_row=len(runs) + 1):
        for cell in row:
            h = heads[cell.column - 1]
            if h == VS_BASE:
                cell.number_format = "+0.00%;-0.00%;0.00%"
            elif h.startswith(("Val MAE", "Val RMSE")):
                cell.number_format = "0.000"
            elif h == "VAR ridge":
                cell.number_format = "0.0E+00"
    for config in {r["config"] for r in runs}:
        rows = [i for i, r in enumerate(runs, 2) if r["config"] == config]
        best = min(rows, key=lambda i: ws.cell(row=i, column=mae_col).value)
        ws.cell(row=best, column=mae_col).fill = BEST_FILL
        ws.cell(row=best, column=mae_col).font = BOLD
    set_widths(ws, [14, 11, 6, 8, 9, 10, 9, 8, 8, 8, 9, 9, 9, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 16, 8, 8, 8, 10, 30])
    ws.cell(row=len(runs) + 3, column=1,
            value="Green = lowest validation MAE for that config. Rank settings by validation only; "
                  "the test sheet is for the final chosen setting.").font = Font(italic=True)


def test_sheet(wb, runs, final_tags):
    ws = wb.create_sheet("Final test")
    heads = ["Tag", "Config", "Seed", "Hidden", "Dropout"]
    for metric in ("MAE", "RMSE", "MAPE %"):
        heads += [f"Test {metric} {MINUTES[h]} min" for h in ALL_HORIZONS] + [f"Test {metric} mean"]
    heads += ["Gate 15 min", "Gate 30 min", "Gate 60 min", "Parameters"]      # the 5-minute gate, when present, is in results.json
    ws.append(heads)
    style_header(ws, len(heads))
    ws.freeze_panes = "D2"
    rows = [r for r in runs if r["tag"] in final_tags]
    for r in sorted(rows, key=lambda r: (r["tag"], r["config"], r["seed"] or 0)):
        t = r["test"]
        line = [r["tag"], r["config"], r["seed"], r["hidden"], r["dropout"]]
        for metric in ("mae", "rmse", "mape"):
            line += [t[h][metric] if h in t else None for h in ALL_HORIZONS] + [t["mean"][metric]]
        line += [r["gate"].get(h) for h in HORIZONS] + [r["params"]]
        ws.append(line)
    for row in ws.iter_rows(min_row=2, max_row=len(rows) + 1, min_col=6, max_col=len(heads) - 4):
        for cell in row:
            cell.number_format = "0.000"
    if not rows:
        ws.cell(row=3, column=1, value="Empty on purpose: fill with --final-tags <tag> once the setting is chosen on "
                                       "validation. Test numbers are read once, for the final setting only.").font = Font(italic=True)
    set_widths(ws, [14, 11, 6, 8, 8] + [11] * 15 + [9, 9, 9, 10])


def comparison_sheet(wb):
    ws = wb.create_sheet("Comparison")
    heads = ["Model", "Code source", "Training recipe", "Epochs run", "Seeds", "Parameters", "Hardware / minutes",
             "MAE 5 min", "MAE 15 min", "MAE 30 min", "MAE 60 min", "RMSE 5 min", "RMSE 15 min", "RMSE 30 min", "RMSE 60 min",
             "MAPE 5 min %", "MAPE 15 min %", "MAPE 30 min %", "MAPE 60 min %", "Notes"]
    ws.append(heads)
    style_header(ws, len(heads))
    ws.freeze_panes = "B2"
    for name in ("HICP none (ours)", "HICP family (ours)", "HICP g12_all (ours)"):
        ws.append([name, "HGP/ (this repository)", "fixed protocol, early stopping on validation MAE"])
    ws.append([])
    ws.cell(row=ws.max_row + 1, column=1,
            value="One row per reference model: same PEMS-BAY split, same masked metrics, same seeds.").font = Font(italic=True)
    set_widths(ws, [24, 26, 34, 10, 8, 11, 18] + [10] * 12 + [30])


def rules_sheet(wb):
    ws = wb.create_sheet("Rules")
    lines = [
        "Protocol for tuning and comparison",
        "1. Choose every setting on the validation split. The test split is read once, for the final setting.",
        "2. Tune with the config 'family' at seed 42; then run none, family and g12_all with the chosen setting at seeds 42, 7 and 123.",
        "3. none, family and g12_all always use identical Part B settings (hidden, dropout) and the same training protocol.",
        "4. g12_all uses graphs built with the same Part A setting as family.",
        "5. The fixed protocol is not tuned: learning rate, weight decay, batch size, epochs, patience, split, loss.",
        "6. Reference models: official code and recipe, or a stated comparable search; same split, same masked metrics, same seeds.",
        "7. Report the whole tuning grid (the Tuning sheet) in the appendix, including settings that did not help.",
        "8. State the epochs run and the hardware for every model. Mark any model that reached its epoch cap before converging.",
        "9. A per-horizon claim needs family to beat g12_all across seeds (paired, sensor-level bootstrap); otherwise report that it did not.",
    ]
    for i, text in enumerate(lines, 1):
        ws.cell(row=i, column=1, value=text)
    ws.cell(row=1, column=1).font = BOLD
    ws.column_dimensions["A"].width = 140


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--results", default=str(Path(__file__).resolve().parents[1] / "results"), help="default: <repo>/results")
    ap.add_argument("--out", default=None, help="workbook path (default: <results>/Comparison horizon.xlsx)")
    ap.add_argument("--final-tags", nargs="*", default=[], help="tags whose test metrics go to the 'Final test' sheet")
    ap.add_argument("--include-smoke", action="store_true")
    args = ap.parse_args()

    out = Path(args.out) if args.out else Path(args.results) / "Comparison horizon.xlsx"
    runs = collect(args.results, args.include_smoke)
    if not runs:
        raise SystemExit(f"no runs found under {args.results}/<tag>/<run>/results.json")

    wb = load_workbook(out) if out.exists() else Workbook()
    for name in OWN_SHEETS:
        if name in wb.sheetnames:
            del wb[name]
    tuning_sheet(wb, runs)
    test_sheet(wb, runs, set(args.final_tags))
    comparison_sheet(wb)
    rules_sheet(wb)
    for name in list(wb.sheetnames):
        ws = wb[name]
        if name not in OWN_SHEETS and ws.max_row == 1 and ws.max_column == 1 and ws["A1"].value is None:
            del wb[name]
    wb._sheets.sort(key=lambda s: OWN_SHEETS.index(s.title) if s.title in OWN_SHEETS else 99)
    wb.save(out)
    print(f"{out}: {len(runs)} runs; tags {sorted({r['tag'] for r in runs})}")


if __name__ == "__main__":
    main()
