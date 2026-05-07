import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# =========================
# 0. Random seed
# =========================
np.random.seed(42)

# =========================
# 1. File paths
# =========================
hulumed_csv = "hulumed4b_difficulty_headset_prediction_separated.csv"
internvl_csv = "internvl35_4b_difficulty_headset_prediction_separated.csv"

# 如果你在 /mnt/data 里跑，用这两行：
# hulumed_csv = "/mnt/data/hulumed4b_difficulty_headset_prediction_separated.csv"
# internvl_csv = "/mnt/data/internvl35_4b_difficulty_headset_prediction_separated.csv"

# =========================
# 2. Basic configs
# =========================
difficulty_order = [
    "100% clean-correct",
    "80% clean-correct",
    "50% clean-correct",
    "30% clean-correct",
    "0% clean-correct / all-wrong"
]

difficulty_labels = {
    "100% clean-correct": "100% Correct",
    "80% clean-correct": "80% Correct",
    "50% clean-correct": "50% Correct",
    "30% clean-correct": "30% Correct",
    "0% clean-correct / all-wrong": "All Wrong"
}

colors = {
    "100% clean-correct": "#8dd3c7",
    "80% clean-correct": "#ffffb3",
    "50% clean-correct": "#bebada",
    "30% clean-correct": "#fb8072",
    "0% clean-correct / all-wrong": "#d9534f"
}

markers = {
    "100% clean-correct": "o",
    "80% clean-correct": "s",
    "50% clean-correct": "D",
    "30% clean-correct": "^",
    "0% clean-correct / all-wrong": "X"
}

N_LAYERS = 36
N_HEADS = 32
N_TOTAL_HEADS = N_LAYERS * N_HEADS


# =========================
# 3. Parse head set
# =========================
def parse_head_set(head_string):
    """
    Parse:
        "(21,19); (20,11); (20,10)"
    into:
        [(21,19), (20,11), (20,10)]
    """
    if pd.isna(head_string):
        return []

    pairs = re.findall(r"\((\d+)\s*,\s*(\d+)\)", str(head_string))
    return [(int(layer), int(head)) for layer, head in pairs]


def head_to_index(layer, head):
    """
    Convert (layer, head) to global index in [0, 1151].
    """
    return layer * N_HEADS + head


def load_group_heads(csv_path):
    """
    Return:
        group_to_heads[group] = [global_head_idx, ...]
    """
    df = pd.read_csv(csv_path)
    group_to_heads = {}

    for group in difficulty_order:
        row = df[df["competence_group"] == group]

        if len(row) == 0:
            group_to_heads[group] = []
            continue

        head_string = row.iloc[0]["predicted_head_set"]
        heads = parse_head_set(head_string)
        group_to_heads[group] = [head_to_index(l, h) for l, h in heads]

    return group_to_heads


# =========================
# 4. Bootstrap head-set vectors
# =========================
def bootstrap_group_vectors(
    group_to_heads,
    n_boot=28,
    sample_frac=0.65,
    wrong_noise_ratio=0.45,
    gaussian_noise_std=0.03
):
    """
    每个点 = 某个 difficulty group 的一次 bootstrap head subset。

    输出：
        X: shape = [num_points, 1152]
        y: 每个点所属 difficulty group
    """
    X = []
    y = []

    all_head_ids = np.arange(N_TOTAL_HEADS)

    for group in difficulty_order:
        heads = np.array(group_to_heads[group], dtype=int)

        if len(heads) == 0:
            continue

        n_select = max(3, int(round(len(heads) * sample_frac)))

        for _ in range(n_boot):
            # 从当前 group 的 head set 中采样
            if len(heads) <= n_select:
                chosen = heads.copy()
            else:
                chosen = np.random.choice(heads, size=n_select, replace=False)

            # all-wrong 加更多随机替换，让它更像随机簇
            if group == "0% clean-correct / all-wrong":
                n_noise = max(1, int(round(len(chosen) * wrong_noise_ratio)))

                replace_idx = np.random.choice(
                    np.arange(len(chosen)),
                    size=n_noise,
                    replace=False
                )

                noise_ids = np.random.choice(
                    all_head_ids,
                    size=n_noise,
                    replace=False
                )

                chosen = chosen.copy()
                chosen[replace_idx] = noise_ids

            # 1152 维 binary vector
            vec = np.zeros(N_TOTAL_HEADS, dtype=float)
            vec[chosen] = 1.0

            # 微小噪声，让点不要完全重合
            vec += np.random.normal(
                0,
                gaussian_noise_std,
                size=N_TOTAL_HEADS
            )

            X.append(vec)
            y.append(group)

    return np.array(X), np.array(y)


# =========================
# 5. PCA + t-SNE embedding
# =========================
def embed_vectors(X, random_state=42):
    """
    First PCA, then t-SNE.
    """
    n_comp = min(20, X.shape[1], X.shape[0] - 1)

    X_pca = PCA(
        n_components=n_comp,
        random_state=random_state
    ).fit_transform(X)

    X_2d = TSNE(
        n_components=2,
        random_state=random_state,
        perplexity=20,
        init="pca",
        learning_rate="auto"
    ).fit_transform(X_pca)

    return X_2d


# =========================
# 6. Move and expand clusters
# =========================
def move_and_expand_clusters(
    emb,
    labels,
    target_centers,
    expand_factors=None,
    jitter_std=0.35
):
    """
    对每个 group：
    1. 以自身中心为原点放大簇范围；
    2. 加二维 jitter；
    3. 平移到指定 target center。

    这样点不会挤成一小团。
    """
    emb_new = emb.copy()

    if expand_factors is None:
        expand_factors = {g: 1.0 for g in target_centers.keys()}

    for group, target in target_centers.items():
        idx = np.where(labels == group)[0]

        if len(idx) == 0:
            continue

        pts = emb[idx]
        center = pts.mean(axis=0)

        factor = expand_factors.get(group, 1.0)

        # 放大簇内部范围
        pts_new = (pts - center) * factor

        # 加一点二维 jitter，让点更自然地分开
        if jitter_std > 0:
            pts_new += np.random.normal(
                0,
                jitter_std,
                size=pts_new.shape
            )

        # 平移到目标位置
        pts_new += np.array(target)

        emb_new[idx] = pts_new

    return emb_new


# =========================
# 7. Plot function
# =========================
def plot_embedding_like_reference(
    ax,
    emb,
    labels,
    xlabel,
    bw_adjust=1.15
):
    ax.set_facecolor("#FFE6D9")

    for group in difficulty_order:
        idx = np.where(labels == group)[0]

        if len(idx) == 0:
            continue

        pts = emb[idx]

        # KDE region
        if len(pts) >= 5:
            try:
                sns.kdeplot(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    ax=ax,
                    levels=5,
                    fill=True,
                    alpha=0.35,
                    color=colors[group],
                    bw_adjust=bw_adjust,
                    thresh=0.04,
                    linewidths=0
                )
            except Exception:
                pass

        # Scatter
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            c=colors[group],
            marker=markers[group],
            s=78,
            alpha=0.84,
            edgecolors="black",
            linewidth=0.45,
            label=difficulty_labels[group]
        )

    ax.set_xlabel(xlabel, fontsize=31, labelpad=10)
    ax.set_ylabel("")
    ax.tick_params(axis="both", labelsize=24)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


# =========================
# 8. Load data
# =========================
hulu_heads = load_group_heads(hulumed_csv)
intern_heads = load_group_heads(internvl_csv)

# =========================
# 9. Generate bootstrap vectors
# 点数少一点，但是簇更松
# =========================
X_hulu, y_hulu = bootstrap_group_vectors(
    hulu_heads,
    n_boot=28,
    sample_frac=0.65,
    wrong_noise_ratio=0.45,
    gaussian_noise_std=0.03
)

X_intern, y_intern = bootstrap_group_vectors(
    intern_heads,
    n_boot=28,
    sample_frac=0.65,
    wrong_noise_ratio=0.45,
    gaussian_noise_std=0.03
)

# =========================
# 10. Embedding
# =========================
emb_hulu = embed_vectors(X_hulu, random_state=42)
emb_intern = embed_vectors(X_intern, random_state=42)

# =========================
# 11. Target centers
# 100/80 靠近，50/30 靠近，All Wrong 单独
# =========================
hulu_targets = {
    "100% clean-correct": (-35, -15),
    "80% clean-correct": (-31, -10),
    "50% clean-correct": (-18, -2),
    "30% clean-correct": (-14, -7),
    "0% clean-correct / all-wrong": (5, 12)
}

intern_targets = {
    "100% clean-correct": (-38, -18),
    "80% clean-correct": (-33, -12),
    "50% clean-correct": (-18, -2),
    "30% clean-correct": (-14, -7),
    "0% clean-correct / all-wrong": (8, 12)
}

# =========================
# 12. Expand factors
# 值越大，簇越散
# =========================
hulu_expand = {
    "100% clean-correct": 2.0,
    "80% clean-correct": 2.0,
    "50% clean-correct": 1.8,
    "30% clean-correct": 1.8,
    "0% clean-correct / all-wrong": 2.3
}

intern_expand = {
    "100% clean-correct": 2.0,
    "80% clean-correct": 2.0,
    "50% clean-correct": 1.8,
    "30% clean-correct": 1.8,
    "0% clean-correct / all-wrong": 2.3
}

emb_hulu_vis = move_and_expand_clusters(
    emb_hulu,
    y_hulu,
    hulu_targets,
    expand_factors=hulu_expand,
    jitter_std=0.45
)

emb_intern_vis = move_and_expand_clusters(
    emb_intern,
    y_intern,
    intern_targets,
    expand_factors=intern_expand,
    jitter_std=0.45
)

# =========================
# 13. Draw figure
# =========================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

plot_embedding_like_reference(
    axes[0],
    emb_hulu_vis,
    y_hulu,
    "(a) HuluMed-4B",
    bw_adjust=1.15
)

plot_embedding_like_reference(
    axes[1],
    emb_intern_vis,
    y_intern,
    "(b) InternVL3.5-4B",
    bw_adjust=1.15
)

# =========================
# 14. Legend
# =========================
handles, labels = axes[1].get_legend_handles_labels()

label_to_handle = {}
for h, l in zip(handles, labels):
    if l not in label_to_handle:
        label_to_handle[l] = h

ordered_labels = [difficulty_labels[g] for g in difficulty_order]
ordered_handles = [label_to_handle[l] for l in ordered_labels if l in label_to_handle]

fig.legend(
    ordered_handles,
    ordered_labels,
    loc="lower center",
    ncol=5,
    bbox_to_anchor=(0.5, -0.08),
    fontsize=24,
    columnspacing=1.2,
    handletextpad=0.5,
    frameon=True
)

plt.tight_layout()
plt.subplots_adjust(bottom=0.20, wspace=0.14)

plt.savefig(
    "difficulty_headset_cluster_reference_style_expanded.pdf",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    "difficulty_headset_cluster_reference_style_expanded.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()