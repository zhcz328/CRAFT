import argparse, json, csv
from pathlib import Path
from collections import defaultdict

def load_round(path: str):
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    top = obj.get("top20", [])
    # normalize fields
    for r in top:
        r["layer"] = int(r["layer"])
        r["head"] = int(r["head"])
        r["mean_abs_effect_reduction"] = float(r["mean_abs_effect_reduction"])
        r["mean_abs_base_change"] = float(r["mean_abs_base_change"])
    return obj, top

def filter_heads(rows, mode, eff_min=None, base_max=None, base_min=None):
    out = []
    for r in rows:
        eff = r["mean_abs_effect_reduction"]
        bc  = r["mean_abs_base_change"]
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

def key(r): return (r["layer"], r["head"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round_files", required=True,
                    help="comma-separated json paths, e.g. r0.json,r1.json,r2.json")
    ap.add_argument("--round_names", default=None,
                    help="comma-separated names (optional), e.g. round0_rise,round1_preplateau,round2_tail")
    ap.add_argument("--topk_per_round", type=int, default=20,
                    help="how many top heads per round to consider (<=20 for your files)")
    ap.add_argument("--mode", default="conflict_specific",
                    choices=["conflict_specific","backbone","custom"])
    ap.add_argument("--eff_min", type=float, default=None)
    ap.add_argument("--base_max", type=float, default=None)
    ap.add_argument("--base_min", type=float, default=None)
    ap.add_argument("--combine", default="union", choices=["union","intersect"])
    ap.add_argument("--out_json", default="selected_heads.json")
    ap.add_argument("--out_csv", default="selected_heads.csv")
    args = ap.parse_args()

    paths = [p.strip() for p in args.round_files.split(",") if p.strip()]
    if args.round_names:
        names = [x.strip() for x in args.round_names.split(",") if x.strip()]
        assert len(names) == len(paths), "round_names length must match round_files"
    else:
        names = [Path(p).stem for p in paths]

    # per-round filtered
    per = {}
    meta = {}
    for name, p in zip(names, paths):
        obj, top = load_round(p)
        meta[name] = {k: obj.get(k) for k in ["scan_layers","n_samples","metrics","weights","position","keep_mode"]}
        top = top[:max(1, min(args.topk_per_round, len(top)))]
        per[name] = filter_heads(top, args.mode, args.eff_min, args.base_max, args.base_min)

    # combine across rounds
    sets = {name: set(key(r) for r in rows) for name, rows in per.items()}
    if args.combine == "union":
        chosen = set().union(*sets.values()) if sets else set()
    else:
        # intersect
        chosen = None
        for s in sets.values():
            chosen = s if chosen is None else (chosen & s)
        chosen = chosen or set()

    # build aggregated view
    # take best eff/base among rounds for each (layer,head), and record which rounds selected it
    agg = {}
    hit_rounds = defaultdict(list)
    for name, rows in per.items():
        for r in rows:
            k = key(r)
            hit_rounds[k].append(name)
            if k not in agg:
                agg[k] = dict(layer=r["layer"], head=r["head"],
                              best_effect_reduction=r["mean_abs_effect_reduction"],
                              best_base_change=r["mean_abs_base_change"])
            else:
                # keep best by higher effect_reduction, tie -> lower base_change
                cur = agg[k]
                better = (r["mean_abs_effect_reduction"] > cur["best_effect_reduction"]) or \
                         (r["mean_abs_effect_reduction"] == cur["best_effect_reduction"] and
                          r["mean_abs_base_change"] < cur["best_base_change"])
                if better:
                    cur["best_effect_reduction"] = r["mean_abs_effect_reduction"]
                    cur["best_base_change"] = r["mean_abs_base_change"]

    selected = []
    for k in sorted(chosen):
        row = agg.get(k, {"layer": k[0], "head": k[1]})
        row["selected_in_rounds"] = hit_rounds.get(k, [])
        # useful scores
        bc = row.get("best_base_change", 0.0)
        er = row.get("best_effect_reduction", 0.0)
        row["conflict_specific_score"] = er / (bc + 1e-6)   # 越大越“高效且不伤基线”
        row["backbone_score"] = er * bc                     # 越大越“有效且伤基线”
        selected.append(row)

    # sort: default by conflict_specific_score desc
    if args.mode == "backbone":
        selected.sort(key=lambda x: (-x["backbone_score"], -x["best_effect_reduction"], x["best_base_change"]))
    else:
        selected.sort(key=lambda x: (-x["conflict_specific_score"], -x["best_effect_reduction"], x["best_base_change"]))

    out = {
        "mode": args.mode,
        "thresholds": {"eff_min": args.eff_min, "base_max": args.base_max, "base_min": args.base_min},
        "combine": args.combine,
        "topk_per_round": args.topk_per_round,
        "rounds": names,
        "round_meta": meta,
        "n_selected": len(selected),
        "selected": selected,
    }
    Path(args.out_json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # csv
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["layer","head","best_effect_reduction","best_base_change",
                    "conflict_specific_score","backbone_score","selected_in_rounds"])
        for r in selected:
            w.writerow([r["layer"], r["head"],
                        f'{r.get("best_effect_reduction",0):.6f}',
                        f'{r.get("best_base_change",0):.6f}',
                        f'{r.get("conflict_specific_score",0):.6f}',
                        f'{r.get("backbone_score",0):.6f}',
                        ",".join(r.get("selected_in_rounds", []))])

    print(f"[OK] selected {len(selected)} heads -> {args.out_json}, {args.out_csv}")

if __name__ == "__main__":
    main()
