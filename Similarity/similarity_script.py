"""
Script for the calculation of frame-by-frame similarity metrics between two sets of probabilities (clean vs noisy).
Saves all metrics per frame in an HDF5 file organized by video, calculates aggregate statistics for video and global,
and generates heatmaps and boxplots for video and global.
"""

import h5py
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ------------------ DEBUG FLAG ------------------
DEBUG = True
if DEBUG:
    debug_dir = "debug_frames"
    os.makedirs(debug_dir, exist_ok=True)

# ------------------ SAFE METRIC HELPERS ------------------
def is_problematic_frame(c, n):
    if np.all(c == 0) or np.all(n == 0):
        return True
    if np.isnan(c).any() or np.isnan(n).any():
        return True
    if np.isinf(c).any() or np.isinf(n).any():
        return True
    if np.var(c) == 0 or np.var(n) == 0:
        return True
    return False

def safe_corr(func, c, n, frame_idx=None, vid=None, name="metric"):
    if is_problematic_frame(c, n):
        if DEBUG:
            np.savez_compressed(
                os.path.join(debug_dir, f"{name}_{vid}_frame{frame_idx}.npz"),
                clean=c, noisy=n
            )
        return np.nan
    try:
        return func(c, n)[0]
    except Exception as e:
        if DEBUG:
            np.savez_compressed(
                os.path.join(debug_dir, f"{name}_err_{vid}_frame{frame_idx}.npz"),
                clean=c, noisy=n
            )
        return np.nan

# ------------------ METRICS ------------------
def cosine_per_frame(a, b, **kwargs):
    return np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)

def pearson_per_frame(a, b, vid=None):
    return np.array([safe_corr(pearsonr, c, n, i, vid, "pearson") for i, (c, n) in enumerate(zip(a, b))])

def euclidean_per_frame(a, b, **kwargs):
    return np.linalg.norm(a - b, axis=1)

def spearman_per_frame(a, b, vid=None):
    return np.array([safe_corr(spearmanr, c, n, i, vid, "spearman") for i, (c, n) in enumerate(zip(a, b))])

def kendall_per_frame(a, b, vid=None):
    return np.array([safe_corr(kendalltau, c, n, i, vid, "kendall") for i, (c, n) in enumerate(zip(a, b))])

def jaccard_topk_per_frame(a, b, k=5, **kwargs):
    out = []
    for x, y in zip(a, b):
        topx = set(np.argsort(x)[-k:][::-1])
        topy = set(np.argsort(y)[-k:][::-1])
        inter = len(topx & topy)
        union = len(topx | topy)
        out.append(inter / union if union > 0 else 0.0)
    return np.array(out)

def dcg(scores, labels, k=None):
    order = np.argsort(scores)[::-1]
    if k is not None:
        order = order[:k]
    gains = (2 ** labels[order] - 1) / np.log2(np.arange(2, 2 + len(order)))
    return gains.sum()

def ndcg_topk_per_frame(a, b, k=5, **kwargs):
    out = []
    for labels, scores in zip(a, b):
        ideal = dcg(labels, labels, k=k)
        actual = dcg(scores, labels, k=k)
        out.append(actual / (ideal + 1e-12))
    return np.array(out)

def rbo_score(list1, list2, p=0.9, k=5):
    set1, set2 = set(), set()
    score = 0.0
    for i in range(k):
        if i < len(list1): set1.add(list1[i])
        if i < len(list2): set2.add(list2[i])
        overlap = len(set1 & set2)
        score += (overlap / (i + 1)) * (p ** i)
    return (1 - p) * score

def rbo_topk_per_frame(a, b, p=0.9, k=5, **kwargs):
    out = []
    for x, y in zip(a, b):
        l1 = list(np.argsort(x)[::-1])
        l2 = list(np.argsort(y)[::-1])
        out.append(rbo_score(l1, l2, p=p, k=k))
    return np.array(out)

# ------------------ FILE PATHS ------------------
hdf5_clean = "/Users/gabriele/Desktop/Magistrale/Explainable_and_trustworthy_AI/progetti/venv/concept_gridlock/Similarity/probs_clip.hdf5"
hdf5_noisy = "/Users/gabriele/Desktop/Magistrale/Explainable_and_trustworthy_AI/progetti/venv/concept_gridlock/Similarity/probs_clip_noisy.hdf5"
hdf5_results = "/Users/gabriele/Desktop/Magistrale/Explainable_and_trustworthy_AI/progetti/venv/concept_gridlock/Similarity/similarity_metrics_clip.hdf5"

# ------------------ METRICS CONFIG ------------------
metrics_config = {
    "cosine": cosine_per_frame,
    "pearson": pearson_per_frame,
    "euclidean": euclidean_per_frame,
    "spearman": spearman_per_frame,
    "kendall": kendall_per_frame,
    "jaccard_top3": lambda a, b, vid=None: jaccard_topk_per_frame(a, b, k=3),
    "jaccard_top5": lambda a, b, vid=None: jaccard_topk_per_frame(a, b, k=5),
    "jaccard_top7": lambda a, b, vid=None: jaccard_topk_per_frame(a, b, k=7),
    "ndcg_top3": lambda a, b, vid=None: ndcg_topk_per_frame(a, b, k=3),
    "ndcg_top5": lambda a, b, vid=None: ndcg_topk_per_frame(a, b, k=5),
    "ndcg_top7": lambda a, b, vid=None: ndcg_topk_per_frame(a, b, k=7),
    "rbo_top3": lambda a, b, vid=None: rbo_topk_per_frame(a, b, p=0.9, k=3),
    "rbo_top5": lambda a, b, vid=None: rbo_topk_per_frame(a, b, p=0.9, k=5),
    "rbo_top7": lambda a, b, vid=None: rbo_topk_per_frame(a, b, p=0.9, k=7),
}

metrics_names = list(metrics_config.keys())

# ------------------ GLOBAL STORAGE ------------------
all_frame_metrics = {m: [] for m in metrics_names}

# ------------------ MAIN LOOP ------------------
with h5py.File(hdf5_clean, "r") as f_clean, \
     h5py.File(hdf5_noisy, "r") as f_noisy, \
     h5py.File(hdf5_results, "w") as f_res:

    for vid in f_clean.keys():
        clean = f_clean[vid][...]
        noisy = f_noisy[vid][...]

        # Compute metrics
        frame_metrics = {}
        for mname, func in metrics_config.items():
            frame_metrics[mname] = func(clean, noisy, vid=vid)

        # Debug NaN counts
        if DEBUG:
            print(f"[DEBUG] Video {vid} NaN counts per metric:")
            for m in metrics_names:
                print(f"  {m}: {np.isnan(frame_metrics[m]).sum()} NaNs")

        # Save per-frame metrics
        grp = f_res.create_group(vid)
        for mname, mvalues in frame_metrics.items():
            grp.create_dataset(mname, data=mvalues, compression="gzip")
            all_frame_metrics[mname].append(mvalues)

        # Aggregate stats per video
        stats = {m: {
            "mean": float(np.nanmean(vals)),
            "median": float(np.nanmedian(vals)),
            "std": float(np.nanstd(vals))
        } for m, vals in frame_metrics.items()}

        dt = np.dtype([("metric", "S32"), ("mean", "f4"), ("median", "f4"), ("std", "f4")])
        stats_arr = np.array(
            [(m.encode(), stats[m]['mean'], stats[m]['median'], stats[m]['std']) for m in metrics_names],
            dtype=dt
        )
        grp.create_dataset("video_stats", data=stats_arr)

        # Print stats
        print(f"\n=== Video {vid} ===")
        for m in metrics_names:
            s = stats[m]
            print(f"{m}: mean={s['mean']:.3f}, median={s['median']:.3f}, std={s['std']:.3f}")

# ------------------ GLOBAL AGGREGATION ------------------
print("\n=== Aggregate statistics on all videos ===")
total_stats = {}
for mname in metrics_names:
    all_frames = np.concatenate(all_frame_metrics[mname])
    total_stats[mname] = {
        "mean": float(np.nanmean(all_frames)),
        "median": float(np.nanmedian(all_frames)),
        "std": float(np.nanstd(all_frames))
    }
    print(f"{mname}: mean={total_stats[mname]['mean']:.3f}, median={total_stats[mname]['median']:.3f}, std={total_stats[mname]['std']:.3f}")

with h5py.File(hdf5_results, "a") as f_res:
    grp = f_res.require_group("total_stats")
    dt = np.dtype([("metric", "S32"), ("mean", "f4"), ("median", "f4"), ("std", "f4")])
    stats_arr = np.array(
        [(m.encode(), total_stats[m]['mean'], total_stats[m]['median'], total_stats[m]['std']) for m in metrics_names],
        dtype=dt
    )
    if "global" in grp:
        del grp["global"]
    grp.create_dataset("global", data=stats_arr)

# ------------------ VISUALIZATION ------------------
all_metrics_matrix = np.vstack([np.concatenate(all_frame_metrics[m]) for m in metrics_names])

plt.figure(figsize=(12, 6))
sns.heatmap(all_metrics_matrix, cmap="viridis", yticklabels=metrics_names)
plt.title("Global Heatmap - all metrics and all frames")
plt.xlabel("Concatenated frames from all videos")
plt.ylabel("Metrics")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
sns.boxplot(data=all_metrics_matrix.T, showfliers=False)
plt.xticks(ticks=np.arange(len(metrics_names)), labels=metrics_names, rotation=45)
plt.title("Global Distribution of Metrics - all videos")
plt.ylabel("Value")
plt.tight_layout()
plt.show()