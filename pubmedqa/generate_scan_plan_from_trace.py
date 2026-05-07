import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def parse_weights(weights_str: str, n: int) -> List[float]:
    if not weights_str:
        return [1.0 / n] * n
    parts = [p.strip() for p in weights_str.split(",") if p.strip()]
    if len(parts) != n:
        raise SystemExit(f"--weights expects {n} comma-separated weights, got {len(parts)}")
    w = [float(x) for x in parts]
    s = sum(w)
    if s == 0:
        raise SystemExit("--weights sum to 0")
    return [x / s for x in w]


def combine_scores(layer_score_mean: Dict[str, List[float]], metrics: List[str], weights: List[float]) -> List[float]:
    n_layers = len(next(iter(layer_score_mean.values())))
    out = [0.0] * n_layers
    for m, w in zip(metrics, weights):
        if m not in layer_score_mean:
            raise SystemExit(f"Metric '{m}' not in trace layer_score_mean keys: {sorted(layer_score_mean.keys())}")
        s = layer_score_mean[m]
        if len(s) != n_layers:
            raise SystemExit(f"Metric '{m}' length mismatch")
        for i in range(n_layers):
            out[i] += w * float(s[i])
    return out


def find_plateau_start(
    scores: List[float],
    plateau_min_score_frac: float,
    plateau_eps: float,
    plateau_min_len: int,
    plateau_slope_eps: float,
) -> Optional[int]:
    if not scores:
        return None
    max_score = max(scores)
    if max_score <= 0:
        return None

    thr = plateau_min_score_frac * max_score
    n = len(scores)
    for s in range(0, n - plateau_min_len + 1):
        window = scores[s : s + plateau_min_len]
        if window[0] < thr:
            continue
        diffs = [window[i + 1] - window[i] for i in range(len(window) - 1)]
        if max(abs(d) for d in diffs) > plateau_eps:
            continue
        mean_slope = (window[-1] - window[0]) / max(1, (plateau_min_len - 1))
        if abs(mean_slope) > plateau_slope_eps:
            continue
        return s
    return None


def topk_rise_layers(scores: List[float], rise_k: int, rise_min_layer: int, rise_max_layer: int) -> List[int]:
    n = len(scores)
    if n <= 1 or rise_k <= 0:
        return []
    if rise_max_layer < 0:
        rmax = n - 1
    else:
        rmax = min(rise_max_layer, n - 1)
    rmin = max(0, min(rise_min_layer, n - 1))

    diffs: List[Tuple[float, int]] = []
    for i in range(0, n - 1):
        if i < rmin or i >= rmax:
            continue
        d = scores[i + 1] - scores[i]
        if d > 0:
            diffs.append((d, i + 1))
    diffs.sort(key=lambda x: x[0], reverse=True)

    chosen = []
    seen = set()
    for _, layer in diffs:
        if layer not in seen:
            seen.add(layer)
            chosen.append(layer)
        if len(chosen) >= rise_k:
            break
    return sorted(chosen)


def build_scan_plan(
    scores: List[float],
    tail_k: int,
    preplateau_width: int,
    plateau_min_score_frac: float,
    plateau_eps: float,
    plateau_min_len: int,
    plateau_slope_eps: float,
    topk_fallback: int,
    rise_k: int,
    rise_min_layer: int,
    rise_max_layer: int,
) -> Dict[str, Any]:
    n_layers = len(scores)
    tail_layers = list(range(max(0, n_layers - tail_k), n_layers))
    rise_layers = topk_rise_layers(scores, rise_k, rise_min_layer, rise_max_layer)

    plateau_start = find_plateau_start(
        scores=scores,
        plateau_min_score_frac=plateau_min_score_frac,
        plateau_eps=plateau_eps,
        plateau_min_len=plateau_min_len,
        plateau_slope_eps=plateau_slope_eps,
    )

    if plateau_start is not None:
        a = max(0, plateau_start - preplateau_width)
        b = min(n_layers - 1, plateau_start + 1)
        plan = {
            "method": "plateau_with_slope_guard",
            "plateau_start": plateau_start,
            "round0_rise": rise_layers,
            "round1_preplateau": list(range(a, b + 1)),
            "round2_tail": tail_layers,
            "tail_k": tail_k,
            "preplateau_width": preplateau_width,
            "plateau_min_score_frac": plateau_min_score_frac,
            "plateau_eps": plateau_eps,
            "plateau_min_len": plateau_min_len,
            "plateau_slope_eps": plateau_slope_eps,
            "topk_fallback": topk_fallback,
        }
        return dedup_rounds(plan)

    idx = sorted(range(n_layers), key=lambda i: scores[i], reverse=True)[:topk_fallback]
    idx = sorted(idx)
    plan = {
        "method": "topk_fallback_no_plateau",
        "plateau_start": None,
        "round0_rise": rise_layers,
        "round1_topk_fallback": idx,
        "round2_tail": tail_layers,
        "tail_k": tail_k,
        "topk_fallback": topk_fallback,
        "plateau_min_score_frac": plateau_min_score_frac,
        "plateau_eps": plateau_eps,
        "plateau_min_len": plateau_min_len,
        "plateau_slope_eps": plateau_slope_eps,
    }
    return dedup_rounds(plan)


def dedup_rounds(scan_plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make round layer lists disjoint by priority:
    round0_rise > round1_preplateau/round1_topk_fallback > round2_tail
    """
    round_keys = ["round0_rise", "round1_preplateau", "round1_topk_fallback", "round2_tail"]
    seen = set()
    for key in round_keys:
        vals = scan_plan.get(key)
        if not vals:
            continue
        clean = []
        for x in vals:
            v = int(x)
            if v in seen:
                continue
            seen.add(v)
            clean.append(v)
        scan_plan[key] = clean
    return scan_plan


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate scan_plan.json from layer-trace trace.json")
    ap.add_argument("--trace", required=True, help="Path to trace.json generated by layer_trace_on_pubmed.py")
    ap.add_argument("--out", required=True, help="Output scan_plan.json path")

    # score source
    ap.add_argument("--plan_metric", default="follow_conflict", help="Single metric from layer_score_mean")
    ap.add_argument("--plan_metrics", default="", help="Comma-separated metrics for weighted plan score")
    ap.add_argument("--weights", default="", help="Comma-separated weights for --plan_metrics")

    # plan hyper-params
    ap.add_argument("--tail_k", type=int, default=6)
    ap.add_argument("--preplateau_width", type=int, default=7)
    ap.add_argument("--plateau_min_score_frac", type=float, default=0.9)
    ap.add_argument("--plateau_eps", type=float, default=0.2)
    ap.add_argument("--plateau_min_len", type=int, default=8)
    ap.add_argument("--plateau_slope_eps", type=float, default=0.05)
    ap.add_argument("--topk_fallback", type=int, default=10)
    ap.add_argument("--rise_k", type=int, default=6)
    ap.add_argument("--rise_min_layer", type=int, default=0)
    ap.add_argument("--rise_max_layer", type=int, default=-1)
    args = ap.parse_args()

    trace_path = Path(args.trace)
    obj = json.loads(trace_path.read_text(encoding="utf-8"))
    layer_score_mean = obj.get("layer_score_mean")
    if not isinstance(layer_score_mean, dict) or not layer_score_mean:
        raise SystemExit("trace file missing layer_score_mean")

    if args.plan_metrics:
        metrics = [x.strip() for x in args.plan_metrics.split(",") if x.strip()]
        if not metrics:
            raise SystemExit("--plan_metrics is empty after parsing")
        weights = parse_weights(args.weights, len(metrics))
        plan_scores = combine_scores(layer_score_mean, metrics, weights)
        plan_source: Dict[str, Any] = {
            "mode": "weighted_metrics",
            "metrics": metrics,
            "weights": {m: float(w) for m, w in zip(metrics, weights)},
        }
    else:
        m = args.plan_metric.strip()
        if m not in layer_score_mean:
            raise SystemExit(f"--plan_metric '{m}' not in trace metrics: {sorted(layer_score_mean.keys())}")
        plan_scores = [float(x) for x in layer_score_mean[m]]
        plan_source = {
            "mode": "single_metric",
            "metric": m,
            "weights": {m: 1.0},
        }

    scan_plan = build_scan_plan(
        scores=plan_scores,
        tail_k=args.tail_k,
        preplateau_width=args.preplateau_width,
        plateau_min_score_frac=args.plateau_min_score_frac,
        plateau_eps=args.plateau_eps,
        plateau_min_len=args.plateau_min_len,
        plateau_slope_eps=args.plateau_slope_eps,
        topk_fallback=args.topk_fallback,
        rise_k=args.rise_k,
        rise_min_layer=args.rise_min_layer,
        rise_max_layer=args.rise_max_layer,
    )

    out = {
        "scheme": obj.get("scheme", "B_conflict_evidence_injection"),
        "position": obj.get("position"),
        "patch_k": obj.get("patch_k"),
        "patch_all_tokens": bool(obj.get("patch_all_tokens", False)),
        "patch_mode": obj.get("patch_mode"),
        "include_context": bool(obj.get("include_context", False)),
        "n_samples": obj.get("n_samples"),
        "n_layers": obj.get("n_layers"),
        "plan_scores_source": plan_source,
        "scan_plan": scan_plan,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] saved -> {out_path}")
    print(f"[OK] method={scan_plan.get('method')} n_layers={len(plan_scores)}")
    if scan_plan.get("round0_rise"):
        print(f"[OK] round0_rise={scan_plan['round0_rise']}")
    if scan_plan.get("round1_preplateau"):
        print(f"[OK] round1_preplateau={scan_plan['round1_preplateau']}")
    if scan_plan.get("round1_topk_fallback"):
        print(f"[OK] round1_topk_fallback={scan_plan['round1_topk_fallback']}")
    if scan_plan.get("round2_tail"):
        print(f"[OK] round2_tail={scan_plan['round2_tail']}")


if __name__ == "__main__":
    main()
