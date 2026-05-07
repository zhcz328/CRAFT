import argparse, json, csv
from pathlib import Path


def load_scan(path: str, source: str = "results", topk: int | None = None):
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if source not in obj:
        if source == "results" and "top20" in obj:
            rows = obj["top20"]
        else:
            raise ValueError(f"Input json missing '{source}' field")
    else:
        rows = obj[source]

    norm = []
    for r in rows:
        item = {
            "layer": int(r["layer"]),
            "head": int(r["head"]),
            "mean_abs_effect_reduction": float(r["mean_abs_effect_reduction"]),
            "mean_abs_base_change": float(r["mean_abs_base_change"]),
            "metric_mean_abs_effect_reduction": {k: float(v) for k, v in r.get("metric_mean_abs_effect_reduction", {}).items()},
            "metric_mean_abs_base_change": {k: float(v) for k, v in r.get("metric_mean_abs_base_change", {}).items()},
        }
        norm.append(item)

    # keep the same default ordering as head_scan output (already sorted), but sort again for safety
    norm.sort(key=lambda r: (-r["mean_abs_effect_reduction"], r["mean_abs_base_change"], r["layer"], r["head"]))

    if topk is not None:
        topk = max(1, min(int(topk), len(norm))) if len(norm) > 0 else 0
        norm = norm[:topk]

    return obj, norm


def filter_heads(rows, mode, eff_min=None, base_max=None, base_min=None):
    out = []
    for r in rows:
        eff = r["mean_abs_effect_reduction"]
        bc = r["mean_abs_base_change"]
        ok = True
        if eff_min is not None:
            ok = ok and (eff >= eff_min)
        if mode == "conflict_specific":
            if base_max is not None:
                ok = ok and (bc <= base_max)
        elif mode == "backbone":
            if base_min is not None:
                ok = ok and (bc >= base_min)
        elif mode == "custom":
            if base_max is not None:
                ok = ok and (bc <= base_max)
            if base_min is not None:
                ok = ok and (bc >= base_min)
        else:
            raise ValueError(mode)
        if ok:
            out.append(r)
    return out


def attach_scores(rows):
    out = []
    for r in rows:
        rr = dict(r)
        bc = rr.get("mean_abs_base_change", 0.0)
        er = rr.get("mean_abs_effect_reduction", 0.0)
        rr["conflict_specific_score"] = er / (bc + 1e-6)
        rr["backbone_score"] = er * bc
        out.append(rr)
    return out


def sort_selected(rows, mode):
    if mode == "backbone":
        rows.sort(key=lambda x: (-x["backbone_score"], -x["mean_abs_effect_reduction"], x["mean_abs_base_change"], x["layer"], x["head"]))
    else:
        rows.sort(key=lambda x: (-x["conflict_specific_score"], -x["mean_abs_effect_reduction"], x["mean_abs_base_change"], x["layer"], x["head"]))
    return rows


def main():
    ap = argparse.ArgumentParser(description="Select heads from a single head_scan_merged_unique_layers.json")
    ap.add_argument("--input_json", required=True, help="path to head_scan_merged_unique_layers.json")
    ap.add_argument("--source", default="results", choices=["results", "top20"],
                    help="read from full results or top20")
    ap.add_argument("--topk", type=int, default=None,
                    help="consider only top-k heads after loading/sorting")
    ap.add_argument("--mode", default="conflict_specific",
                    choices=["conflict_specific", "backbone", "custom"])
    ap.add_argument("--eff_min", type=float, default=None)
    ap.add_argument("--base_max", type=float, default=None)
    ap.add_argument("--base_min", type=float, default=None)
    ap.add_argument("--out_json", default="selected_heads_merged_unique_layers.json")
    ap.add_argument("--out_csv", default="selected_heads_merged_unique_layers.csv")
    args = ap.parse_args()

    obj, rows = load_scan(args.input_json, source=args.source, topk=args.topk)
    selected = filter_heads(rows, args.mode, args.eff_min, args.base_max, args.base_min)
    selected = attach_scores(selected)
    selected = sort_selected(selected, args.mode)

    out = {
        "mode": args.mode,
        "thresholds": {
            "eff_min": args.eff_min,
            "base_max": args.base_max,
            "base_min": args.base_min,
        },
        "input_json": args.input_json,
        "source": args.source,
        "topk": args.topk,
        "meta": {
            "round": obj.get("round"),
            "scan_layers": obj.get("scan_layers"),
            "n_samples": obj.get("n_samples"),
            "n_layers": obj.get("n_layers"),
            "n_heads": obj.get("n_heads"),
            "metrics": obj.get("metrics"),
            "weights": obj.get("weights"),
            "position": obj.get("position"),
            "keep_mode": obj.get("keep_mode"),
            "trace_mode": obj.get("trace_mode"),
            "model": obj.get("model"),
        },
        "n_candidates": len(rows),
        "n_selected": len(selected),
        "selected": selected,
    }
    Path(args.out_json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "layer", "head", "mean_abs_effect_reduction", "mean_abs_base_change",
            "conflict_specific_score", "backbone_score",
            "metric_mean_abs_effect_reduction", "metric_mean_abs_base_change"
        ])
        for r in selected:
            w.writerow([
                r["layer"], r["head"],
                f'{r.get("mean_abs_effect_reduction", 0):.6f}',
                f'{r.get("mean_abs_base_change", 0):.6f}',
                f'{r.get("conflict_specific_score", 0):.6f}',
                f'{r.get("backbone_score", 0):.6f}',
                json.dumps(r.get("metric_mean_abs_effect_reduction", {}), ensure_ascii=False),
                json.dumps(r.get("metric_mean_abs_base_change", {}), ensure_ascii=False),
            ])

    print(f"[OK] selected {len(selected)} heads -> {args.out_json}, {args.out_csv}")


if __name__ == "__main__":
    main()
