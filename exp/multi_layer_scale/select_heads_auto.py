#!/usr/bin/env python3
import argparse
import csv
import json
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
        norm.append(
            {
                "layer": int(r["layer"]),
                "head": int(r["head"]),
                "mean_abs_effect_reduction": float(r["mean_abs_effect_reduction"]),
                "mean_abs_base_change": float(r["mean_abs_base_change"]),
                "metric_mean_abs_effect_reduction": {
                    k: float(v) for k, v in r.get("metric_mean_abs_effect_reduction", {}).items()
                },
                "metric_mean_abs_base_change": {
                    k: float(v) for k, v in r.get("metric_mean_abs_base_change", {}).items()
                },
            }
        )

    norm.sort(key=lambda r: (-r["mean_abs_effect_reduction"], r["mean_abs_base_change"], r["layer"], r["head"]))
    if topk is not None:
        topk = max(1, min(int(topk), len(norm))) if norm else 0
        norm = norm[:topk]
    return obj, norm


def filter_heads(rows, mode, eff_min=None, eff_max=None, base_max=None, base_min=None):
    out = []
    for r in rows:
        eff = float(r["mean_abs_effect_reduction"])
        bc = float(r["mean_abs_base_change"])
        ok = True
        if eff_min is not None:
            ok = ok and eff >= eff_min
        if eff_max is not None:
            ok = ok and eff <= eff_max
        if mode == "conflict_specific":
            if base_max is not None:
                ok = ok and bc <= base_max
        elif mode == "backbone":
            if base_min is not None:
                ok = ok and bc >= base_min
        elif mode == "custom":
            if base_max is not None:
                ok = ok and bc <= base_max
            if base_min is not None:
                ok = ok and bc >= base_min
        else:
            raise ValueError(mode)
        if ok:
            out.append(r)
    return out


def summarize_selection(rows):
    if not rows:
        return {
            "count": 0,
            "mean_effect_reduction": 0.0,
            "mean_base_change": 0.0,
            "max_effect_reduction": 0.0,
            "min_effect_reduction": 0.0,
            "max_base_change": 0.0,
            "min_base_change": 0.0,
        }
    effect_vals = [float(r["mean_abs_effect_reduction"]) for r in rows]
    base_vals = [float(r["mean_abs_base_change"]) for r in rows]
    return {
        "count": len(rows),
        "mean_effect_reduction": float(sum(effect_vals) / len(effect_vals)),
        "mean_base_change": float(sum(base_vals) / len(base_vals)),
        "max_effect_reduction": float(max(effect_vals)),
        "min_effect_reduction": float(min(effect_vals)),
        "max_base_change": float(max(base_vals)),
        "min_base_change": float(min(base_vals)),
    }


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


def auto_detect_thresholds(rows, mode, target_min, target_max, eff_floor, eff_cap, base_max_cap, base_min_floor):
    search_rows = []
    for r in rows:
        eff = float(r["mean_abs_effect_reduction"])
        base = float(r["mean_abs_base_change"])
        if eff <= max(0.0, float(eff_floor)):
            continue
        if eff_cap is not None and eff > float(eff_cap):
            continue
        if mode in {"conflict_specific", "custom"} and base_max_cap is not None and base > float(base_max_cap):
            continue
        if mode in {"backbone", "custom"} and base_min_floor is not None and base < float(base_min_floor):
            continue
        search_rows.append(r)
    if not search_rows:
        return {
            "eff_min": None,
            "eff_max": eff_cap,
            "base_max": base_max_cap if mode != "backbone" else None,
            "base_min": base_min_floor if mode == "backbone" else None,
            "count": 0,
            "summary": summarize_selection([]),
            "selected_directly": True,
        }

    if len(search_rows) <= int(target_max):
        return {
            "eff_min": None,
            "eff_max": eff_cap,
            "base_max": base_max_cap if mode != "backbone" else None,
            "base_min": base_min_floor if mode == "backbone" else None,
            "count": len(search_rows),
            "summary": summarize_selection(search_rows),
            "selected_directly": True,
        }

    target_mid = (float(target_min) + float(target_max)) / 2.0
    eff_candidates = sorted({max(float(eff_floor), float(r["mean_abs_effect_reduction"])) for r in search_rows})
    if mode == "backbone":
        base_candidates = sorted({float(r["mean_abs_base_change"]) for r in search_rows}, reverse=True)
    else:
        base_candidates = sorted({float(r["mean_abs_base_change"]) for r in search_rows})
        if base_max_cap is not None:
            base_candidates = [b for b in base_candidates if b <= float(base_max_cap)]
    if not eff_candidates or not base_candidates:
        raise ValueError("No threshold candidates available.")

    best = None
    for eff_thr in eff_candidates:
        for base_thr in base_candidates:
            if mode == "backbone":
                selected = filter_heads(rows, mode, eff_min=eff_thr, eff_max=eff_cap, base_min=base_thr)
            else:
                selected = filter_heads(
                    rows,
                    mode,
                    eff_min=eff_thr,
                    eff_max=eff_cap,
                    base_max=base_thr,
                    base_min=base_min_floor if mode == "custom" else None,
                )
            stats = summarize_selection(selected)
            count = stats["count"]
            if count <= 0:
                continue
            in_target = int(target_min <= count <= target_max)
            rank_key = (
                -in_target,
                abs(count - target_mid),
                -stats["mean_effect_reduction"],
                stats["mean_base_change"],
                -float(eff_thr),
                float(base_thr),
            )
            candidate = {
                "eff_min": float(eff_thr),
                "eff_max": None if eff_cap is None else float(eff_cap),
                "base_max": float(base_thr) if mode != "backbone" else None,
                "base_min": float(base_thr) if mode == "backbone" else base_min_floor,
                "count": count,
                "summary": stats,
                "selected_directly": False,
            }
            if best is None or rank_key < best["rank_key"]:
                best = {"rank_key": rank_key, "candidate": candidate}
    if best is None:
        return {
            "eff_min": None,
            "eff_max": eff_cap,
            "base_max": base_max_cap if mode != "backbone" else None,
            "base_min": base_min_floor if mode == "backbone" else None,
            "count": len(search_rows),
            "summary": summarize_selection(search_rows),
            "selected_directly": True,
        }
    return best["candidate"]


def main():
    ap = argparse.ArgumentParser(description="Standalone auto head selector for multi-layer-scale experiments.")
    ap.add_argument("--input_json", required=True)
    ap.add_argument("--source", default="results", choices=["results", "top20"])
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--mode", default="conflict_specific", choices=["conflict_specific", "backbone", "custom"])
    ap.add_argument("--target_min", type=int, default=6)
    ap.add_argument("--target_max", type=int, default=7)
    ap.add_argument("--eff_floor", type=float, default=1e-6)
    ap.add_argument("--eff_cap", type=float, default=0.2)
    ap.add_argument("--base_max_cap", type=float, default=0.2)
    ap.add_argument("--base_min_floor", type=float, default=None)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    obj, rows = load_scan(args.input_json, source=args.source, topk=args.topk)
    auto = auto_detect_thresholds(
        rows=rows,
        mode=args.mode,
        target_min=args.target_min,
        target_max=args.target_max,
        eff_floor=args.eff_floor,
        eff_cap=args.eff_cap,
        base_max_cap=args.base_max_cap,
        base_min_floor=args.base_min_floor,
    )
    if auto.get("selected_directly"):
        selected = filter_heads(
            rows,
            args.mode,
            eff_min=args.eff_floor,
            eff_max=args.eff_cap,
            base_max=args.base_max_cap if args.mode != "backbone" else None,
            base_min=args.base_min_floor if args.mode == "backbone" else None,
        )
    else:
        selected = filter_heads(
            rows,
            args.mode,
            eff_min=auto["eff_min"],
            eff_max=auto["eff_max"],
            base_max=auto["base_max"],
            base_min=auto["base_min"],
        )
    selected = sort_selected(attach_scores(selected), args.mode)

    out = {
        "mode": args.mode,
        "thresholds": {
            "eff_min": auto["eff_min"],
            "eff_max": auto["eff_max"],
            "base_max": auto["base_max"],
            "base_min": auto["base_min"],
        },
        "auto_search": {
            "target_min": args.target_min,
            "target_max": args.target_max,
            "eff_floor": args.eff_floor,
            "eff_cap": args.eff_cap,
            "base_max_cap": args.base_max_cap,
            "base_min_floor": args.base_min_floor,
            "selected_directly": bool(auto.get("selected_directly")),
            "chosen_summary": summarize_selection(selected),
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
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "layer",
                "head",
                "mean_abs_effect_reduction",
                "mean_abs_base_change",
                "conflict_specific_score",
                "backbone_score",
            ]
        )
        for r in selected:
            w.writerow(
                [
                    r["layer"],
                    r["head"],
                    f'{r["mean_abs_effect_reduction"]:.6f}',
                    f'{r["mean_abs_base_change"]:.6f}',
                    f'{r["conflict_specific_score"]:.6f}',
                    f'{r["backbone_score"]:.6f}',
                ]
            )
    print(json.dumps({"out_json": args.out_json, "out_csv": args.out_csv, "n_selected": len(selected)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
