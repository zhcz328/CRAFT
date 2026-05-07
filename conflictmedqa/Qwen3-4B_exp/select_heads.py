import argparse
import csv
import json
import math
from pathlib import Path


DEFAULT_SUMMARY_PATH = "/home/zengjiaqi/icl/interp/logit_lens/conflictmedqa/Qwen3-4B_exp/result/headscan_rounds_top30_inf/head_scan_summary.json"
DEFAULT_OUT_DIR = Path("/home/zengjiaqi/icl/interp/logit_lens/conflictmedqa/Qwen3-4B_exp/result/headscan_rounds_top30_inf")
DEFAULT_MODE = "B"
DEFAULT_ER_QUANTILE = 0.75
DEFAULT_BC_QUANTILE = 0.25
DEFAULT_TOPK_FALLBACK = 50
DEFAULT_ER_MIN = 3.5
DEFAULT_BC_MAX = 1.5
DEFAULT_PRINT_TOPN = 30
EPS = 1e-6


def score_cs(er, bc):
    return er / (bc + EPS)


def quantile(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    if q <= 0:
        return xs[0]
    if q >= 1:
        return xs[-1]
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    w = pos - lo
    return xs[lo] * (1 - w) + xs[hi] * w


def load_heads_from_summary(summary_path: str):
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    heads = []

    def ingest_item(item, source_tag):
        if not isinstance(item, dict):
            return
        if "layer" not in item or "head" not in item:
            return
        er = item.get("mean_abs_effect_reduction")
        bc = item.get("mean_abs_base_delta_change")
        if er is None or bc is None:
            return
        heads.append(
            {
                "layer": int(item["layer"]),
                "head": int(item["head"]),
                "er": float(er),
                "bc": float(bc),
                "source": source_tag,
            }
        )

    round_outputs = data.get("round_outputs") or []
    for round_rec in round_outputs:
        if not isinstance(round_rec, dict):
            continue
        round_name = round_rec.get("round") or round_rec.get("name") or "round"
        round_result = round_rec.get("result") or {}
        top = round_result.get("top20") or round_result.get("top") or round_result.get("heads") or []
        if isinstance(top, dict):
            top = list(top.values())
        for item in top:
            ingest_item(item, f"round_outputs:{round_name}")

    prs = data.get("per_round_summaries") or data.get("rounds") or []
    if isinstance(prs, dict):
        prs = list(prs.values())
    for round_rec in prs:
        if not isinstance(round_rec, dict):
            continue
        round_name = round_rec.get("round_name") or round_rec.get("name") or "round"
        top = round_rec.get("top20") or round_rec.get("top") or round_rec.get("heads") or []
        if isinstance(top, dict):
            top = list(top.values())
        for item in top:
            ingest_item(item, f"per_round:{round_name}")

    for key in ["all_heads", "candidates", "head_stats", "heads", "conflict_specific", "backbone"]:
        arr = data.get(key)
        if isinstance(arr, list):
            for item in arr:
                ingest_item(item, f"summary:{key}")

    best = {}
    for head in heads:
        key = (head["layer"], head["head"])
        current = best.get(key)
        if current is None or head["er"] > current["er"] or (
            head["er"] == current["er"] and head["bc"] < current["bc"]
        ):
            best[key] = head

    deduped = list(best.values())
    for head in deduped:
        head["cs_score"] = score_cs(head["er"], head["bc"])
    return deduped


def main():
    ap = argparse.ArgumentParser(description="Select conflict-specific heads from a head-scan summary.")
    ap.add_argument("--summary_path", default=DEFAULT_SUMMARY_PATH)
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--mode", default=DEFAULT_MODE, choices=["A", "B", "a", "b"])
    ap.add_argument("--er_quantile", type=float, default=DEFAULT_ER_QUANTILE)
    ap.add_argument("--bc_quantile", type=float, default=DEFAULT_BC_QUANTILE)
    ap.add_argument("--topk_fallback", type=int, default=DEFAULT_TOPK_FALLBACK)
    ap.add_argument("--er_min", type=float, default=DEFAULT_ER_MIN)
    ap.add_argument("--bc_max", type=float, default=DEFAULT_BC_MAX)
    ap.add_argument("--print_topn", type=int, default=DEFAULT_PRINT_TOPN)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "selected_heads.json"
    out_csv = out_dir / "selected_heads.csv"

    heads = load_heads_from_summary(args.summary_path)
    if not heads:
        raise RuntimeError(f"No head records could be parsed from {args.summary_path}")

    ers = [head["er"] for head in heads]
    bcs = [head["bc"] for head in heads]

    if args.mode.upper() == "A":
        er_thr = quantile(ers, args.er_quantile)
        bc_thr = quantile(bcs, args.bc_quantile)
        selected = [head for head in heads if head["er"] >= er_thr and head["bc"] <= bc_thr]
        if len(selected) < min(10, len(heads)):
            selected = sorted(heads, key=lambda x: (-x["cs_score"], -x["er"], x["bc"]))[
                : min(args.topk_fallback, len(heads))
            ]
        meta = {
            "mode": "A_quantile",
            "er_quantile": args.er_quantile,
            "bc_quantile": args.bc_quantile,
            "er_threshold": er_thr,
            "bc_threshold": bc_thr,
            "fallback_topk": args.topk_fallback,
            "total_candidates": len(heads),
            "selected": len(selected),
        }
    else:
        selected = [head for head in heads if head["er"] >= args.er_min and head["bc"] <= args.bc_max]
        selected = sorted(selected, key=lambda x: (-x["cs_score"], -x["er"], x["bc"]))
        meta = {
            "mode": "B_fixed",
            "er_min": args.er_min,
            "bc_max": args.bc_max,
            "total_candidates": len(heads),
            "selected": len(selected),
        }

    selected = sorted(selected, key=lambda x: (-x["cs_score"], -x["er"], x["bc"]))

    payload = {"meta": meta, "selected_heads": selected}
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "head", "er", "bc", "cs_score", "source"])
        for head in selected:
            writer.writerow([head["layer"], head["head"], head["er"], head["bc"], head["cs_score"], head["source"]])

    print("==== selection meta ====")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"\n==== top {min(args.print_topn, len(selected))} selected heads ====")
    for idx, head in enumerate(selected[: args.print_topn], 1):
        print(
            f"{idx:02d}. L{head['layer']:>3} H{head['head']:>3}  "
            f"ER={head['er']:.4f}  BC={head['bc']:.4f}  score={head['cs_score']:.4f}  ({head['source']})"
        )

    print(f"\nWrote:\n- {out_json}\n- {out_csv}")


if __name__ == "__main__":
    main()
