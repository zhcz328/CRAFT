import json
import math
import csv
from pathlib import Path

# ====== 配置区：你只需要改这里 ======
SUMMARY_PATH = "/home/zengjiaqi/icl/interp/logit_lens/conflictmedqa/Qwen3-4B_exp/result/headscan_rounds_top30_inf/head_scan_summary.json"

# 筛选策略：二选一（推荐 A）
MODE = "B"  # "A"=分位数筛选(自适应)；"B"=固定阈值筛选

# A: 分位数筛选（自适应，不容易筛空）
ER_QUANTILE = 0.75   # ER 取 top 25%
BC_QUANTILE = 0.25   # BC 取 bottom 25%
TOPK_FALLBACK = 50   # 如果 A 筛出来太少，就按综合分数取前 TOPK_FALLBACK

# B: 固定阈值筛选（你可以自己设）
ER_MIN = 3.5
BC_MAX = 1.5

# 综合分数（越大越好）：ER 大、BC 小
# 你也可以改成 ER - lam*BC
EPS = 1e-6
def score_cs(er, bc):
    return er / (bc + EPS)

# 输出文件
OUT_DIR = Path("/home/zengjiaqi/icl/interp/logit_lens/conflictmedqa/Qwen3-4B_exp/result/headscan_rounds_top30_inf")
OUT_JSON = OUT_DIR / "selected_heads.json"
OUT_CSV  = OUT_DIR / "selected_heads.csv"
PRINT_TOPN = 30
# ===================================


def quantile(xs, q):
    """简单分位数（线性插值）"""
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
    data = json.load(open(summary_path, "r", encoding="utf-8"))

    # 兼容不同字段命名：尽量从 summary 里“所有出现过的 head 记录”收集
    # 常见来源：per_round_summaries -> top20；或 all_heads / candidates 等
    heads = []

    def ingest_item(item, source_tag):
        if not isinstance(item, dict):
            return
        # 需要至少 layer/head 和 ER/BC
        if "layer" not in item or "head" not in item:
            return
        er = item.get("mean_abs_effect_reduction")
        bc = item.get("mean_abs_base_delta_change")
        if er is None or bc is None:
            return
        heads.append({
            "layer": int(item["layer"]),
            "head": int(item["head"]),
            "er": float(er),
            "bc": float(bc),
            "source": source_tag,
        })

    # 1) round summaries 里抓 top20
    prs = data.get("per_round_summaries") or data.get("rounds") or []
    if isinstance(prs, dict):
        prs = list(prs.values())

    for r in prs:
        if not isinstance(r, dict):
            continue
        rname = r.get("round_name") or r.get("name") or "round"
        top = r.get("top20") or r.get("top") or r.get("heads") or []
        if isinstance(top, dict):
            top = list(top.values())
        for item in top:
            ingest_item(item, f"per_round:{rname}")

    # 2) 兜底：summary 里若存在 all_heads/candidates/head_stats 之类
    for k in ["all_heads", "candidates", "head_stats", "heads"]:
        arr = data.get(k)
        if isinstance(arr, list):
            for item in arr:
                ingest_item(item, f"summary:{k}")

    # 去重：同一 (layer,head) 可能出现多次，取“更好”的（er 更大；若相等 bc 更小）
    best = {}
    for h in heads:
        key = (h["layer"], h["head"])
        if key not in best:
            best[key] = h
        else:
            cur = best[key]
            if (h["er"] > cur["er"]) or (h["er"] == cur["er"] and h["bc"] < cur["bc"]):
                best[key] = h

    heads = list(best.values())
    # 加分数
    for h in heads:
        h["cs_score"] = score_cs(h["er"], h["bc"])
    return heads


def main():
    heads = load_heads_from_summary(SUMMARY_PATH)
    if not heads:
        raise RuntimeError("没有从 head_scan_summary.json 里解析到 head 记录。请把文件内容/结构发我，我给你适配字段。")

    ers = [h["er"] for h in heads]
    bcs = [h["bc"] for h in heads]

    if MODE.upper() == "A":
        er_thr = quantile(ers, ER_QUANTILE)
        bc_thr = quantile(bcs, BC_QUANTILE)

        selected = [h for h in heads if (h["er"] >= er_thr and h["bc"] <= bc_thr)]

        # 如果筛太少，就按分数回退取前 TOPK_FALLBACK
        if len(selected) < min(10, len(heads)):
            heads_sorted = sorted(heads, key=lambda x: (-x["cs_score"], -x["er"], x["bc"]))
            selected = heads_sorted[:min(TOPK_FALLBACK, len(heads_sorted))]

        meta = {
            "mode": "A_quantile",
            "er_quantile": ER_QUANTILE,
            "bc_quantile": BC_QUANTILE,
            "er_threshold": er_thr,
            "bc_threshold": bc_thr,
            "fallback_topk": TOPK_FALLBACK,
            "total_candidates": len(heads),
            "selected": len(selected),
        }

    else:
        selected = [h for h in heads if (h["er"] >= ER_MIN and h["bc"] <= BC_MAX)]
        selected = sorted(selected, key=lambda x: (-x["cs_score"], -x["er"], x["bc"]))
        meta = {
            "mode": "B_fixed",
            "er_min": ER_MIN,
            "bc_max": BC_MAX,
            "total_candidates": len(heads),
            "selected": len(selected),
        }

    # 排序输出：cs_score 优先
    selected = sorted(selected, key=lambda x: (-x["cs_score"], -x["er"], x["bc"]))

    # 写 JSON
    out = {"meta": meta, "selected_heads": selected}
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # 写 CSV
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["layer", "head", "er", "bc", "cs_score", "source"])
        for h in selected:
            w.writerow([h["layer"], h["head"], h["er"], h["bc"], h["cs_score"], h["source"]])

    # 打印前 N
    print("==== selection meta ====")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"\n==== top {min(PRINT_TOPN, len(selected))} selected heads ====")
    for i, h in enumerate(selected[:PRINT_TOPN], 1):
        print(f"{i:02d}. L{h['layer']:>3} H{h['head']:>3}  ER={h['er']:.4f}  BC={h['bc']:.4f}  score={h['cs_score']:.4f}  ({h['source']})")

    print(f"\nWrote:\n- {OUT_JSON}\n- {OUT_CSV}")


if __name__ == "__main__":
    main()
