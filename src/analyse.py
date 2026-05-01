"""
Stage 6 – analyse.py
─────────────────────
Full statistical analysis pipeline covering Sections 5–9 of the project brief.

Sections implemented
────────────────────
5.  Descriptive Statistics
    5.1 Acoustic features (mean/sd/IQR/CV, variance decomposition, plots)
    5.2 Neural representations (PCA/UMAP plots, between-class variance ratio,
        cosine similarity within/between phoneme)
    5.3 Cross-representation RSM + Mantel test

6.  Statistical Tests
    6.1 Group comparisons (L1 vs L2 t-test/Mann-Whitney, BH correction;
        permutation test on neural cosine distance)
    6.2 Inter-phoneme distances (Euclidean, Mahalanobis, cosine distance
        matrices; Mantel test; bootstrap CIs; nearest-centroid LOSO classifier)

7.  Linear Mixed-Effects Models
    7.1/7.2 LME for acoustic F1/F2 and neural PC1-5
    7.3 Model building (null → main → full → extended → random-slope)
    7.4 Marginal/conditional R² comparison

8.  Confidence Intervals and ROPE
    8.1/8.2 Forest plots for acoustic and neural contrasts
    8.3/8.4 ROPE classification table

9.  Hierarchical Clustering
    9.1 French oral vowels
    9.2 Consonants vs. vowels
    9.3 Speaker clustering
    9.4 Silhouette / dendrogram / linguistic coherence

Outputs
───────
results/figures/          PNG figures (numbered by section)
results/stats_summary.json  machine-readable summary metrics

Usage (called by DVC):
    python src/analyse.py
"""

import json
import logging
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import (
    adjusted_rand_score,
    silhouette_score,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
from sklearn.preprocessing import LabelEncoder
import yaml

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── French oral vowel inventory (canonical) ───────────────────────────────────
ORAL_VOWELS = ["a", "e", "i", "o", "u", "y", "ø", "œ", "ɛ", "ɔ", "ə", "ɑ"]
NASAL_VOWELS = ["ɑ̃", "ɛ̃", "œ̃", "ɔ̃"]
ALL_VOWELS = ORAL_VOWELS + NASAL_VOWELS

# IPA vowel trapezoid ground-truth partitions for ARI evaluation
FRONT_VOWELS  = {"i", "e", "ɛ", "a", "y", "ø", "œ"}
BACK_VOWELS   = {"u", "o", "ɔ", "ɑ"}
HIGH_VOWELS   = {"i", "y", "u"}
MID_VOWELS    = {"e", "ø", "o", "ɛ", "œ", "ɔ", "ə"}
LOW_VOWELS    = {"a", "ɑ"}

# Selected consonant classes for Section 9.2
TARGET_CONSONANTS = ["p", "t", "k", "b", "d", "s", "z", "ʃ", "ʒ", "m", "n", "l", "ʁ"]


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def load_params(path="params.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 - cosine_similarity between two 1-D vectors."""
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 1.0
    return float(1.0 - np.dot(a, b) / norm)


def bh_correction(pvalues: list[float]) -> np.ndarray:
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values."""
    n = len(pvalues)
    if n == 0:
        return np.array([])
    idx = np.argsort(pvalues)
    adj = np.empty(n)
    for rank, i in enumerate(idx, 1):
        adj[i] = pvalues[i] * n / rank
    # Enforce monotonicity
    for k in range(n - 2, -1, -1):
        adj[idx[k]] = min(adj[idx[k]], adj[idx[k + 1]])
    return np.minimum(adj, 1.0)


def mantel_test(D1: np.ndarray, D2: np.ndarray, n_perm: int = 999) -> tuple[float, float]:
    """
    Mantel test: rank correlation between upper triangles of two distance matrices.
    Returns (r, p_value).
    """
    triu = np.triu_indices(D1.shape[0], k=1)
    x = D1[triu]
    y = D2[triu]
    r_obs, _ = stats.spearmanr(x, y)
    count = 0
    rng = np.random.default_rng(42)
    for _ in range(n_perm):
        perm = rng.permutation(len(x))
        r_perm, _ = stats.spearmanr(x[perm], y)
        if abs(r_perm) >= abs(r_obs):
            count += 1
    p = (count + 1) / (n_perm + 1)
    return float(r_obs), float(p)


def bootstrap_ci(data: np.ndarray, stat_fn, B: int = 2000, alpha: float = 0.05):
    """Percentile bootstrap CI for stat_fn(data)."""
    rng = np.random.default_rng(42)
    stats_boot = [stat_fn(rng.choice(data, size=len(data), replace=True))
                  for _ in range(B)]
    lo = np.percentile(stats_boot, 100 * alpha / 2)
    hi = np.percentile(stats_boot, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def savefig(fig, path: Path, tight=True):
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved {path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data(params: dict):
    """Load all available data files. Neural files may not exist yet."""
    norm_dir  = Path(params["normalise"]["output_dir"])
    proc_dir  = Path(params["data"]["processed_dir"])

    # Acoustic (always available after local stages)
    ac_path = Path(params["normalise"]["lobanov_output"])
    df_ac = pd.read_csv(ac_path, encoding="utf-8") if ac_path.exists() else None

    # Manifest (for joining)
    manifest = pd.read_csv(proc_dir / "phonemes.csv", encoding="utf-8")

    # Neural (may not exist if Kaggle hasn't run yet)
    def load_npz(path):
        if Path(path).exists():
            d = np.load(path, allow_pickle=True)
            meta = json.loads(str(d["meta_json"]))
            return d, pd.DataFrame(meta)
        return None, None

    xlsr_npz, df_xlsr_meta    = load_npz(norm_dir / "features_xlsr_pca.npz")
    whisper_npz, df_wh_meta   = load_npz(norm_dir / "features_whisper_pca.npz")

    return {
        "acoustic":    df_ac,
        "manifest":    manifest,
        "xlsr_npz":    xlsr_npz,
        "whisper_npz": whisper_npz,
        "xlsr_meta":   df_xlsr_meta,
        "whisper_meta": df_wh_meta,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 – Descriptive Statistics
# ─────────────────────────────────────────────────────────────────────────────

def section5_acoustic(df: pd.DataFrame, fig_dir: Path) -> dict:
    log.info("Section 5.1: Acoustic descriptive statistics …")
    results = {}

    vowel_df = df[df["phoneme_base"].isin(ORAL_VOWELS)].copy()

    # ── 5.1a Summary statistics per phoneme per group ─────────────────────────
    summary_rows = []
    for phoneme, grp in vowel_df.groupby("phoneme_base"):
        for group_label, g in grp.groupby("l1_status"):
            for f in ["F1_lob", "F2_lob"]:
                vals = g[f].dropna()
                if len(vals) == 0:
                    continue
                summary_rows.append({
                    "phoneme": phoneme,
                    "group":   group_label,
                    "feature": f,
                    "mean":    round(vals.mean(), 4),
                    "median":  round(vals.median(), 4),
                    "sd":      round(vals.std(), 4),
                    "iqr":     round(vals.quantile(0.75) - vals.quantile(0.25), 4),
                    "cv":      round(vals.std() / abs(vals.mean()), 4) if vals.mean() != 0 else float("nan"),
                    "n":       len(vals),
                })
    results["acoustic_summary"] = summary_rows

    # ── 5.1b Variance decomposition for F1 ───────────────────────────────────
    var_decomp = {}
    for phoneme, grp in vowel_df.groupby("phoneme_base"):
        vals = grp["F1_lob"].dropna()
        if len(vals) < 5:
            continue
        total_var = vals.var()
        speaker_means = grp.groupby("speaker")["F1_lob"].mean()
        inter_var = speaker_means.var() if len(speaker_means) > 1 else 0.0
        residual  = total_var - inter_var
        var_decomp[phoneme] = {
            "total":      round(float(total_var), 4),
            "inter_spk":  round(float(inter_var), 4),
            "residual":   round(float(max(residual, 0)), 4),
        }
    results["variance_decomposition_F1"] = var_decomp

    # ── 5.1c Vowel chart: F1 vs F2 centroids with 95% CI ellipses ────────────
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = {"L1": "#2196F3", "L2": "#F44336"}
    markers = {"L1": "o", "L2": "^"}

    for (phoneme, l1), grp in vowel_df.groupby(["phoneme_base", "l1_status"]):
        f1 = grp["F1_lob"].dropna()
        f2 = grp["F2_lob"].dropna()
        if len(f1) < 3:
            continue
        # Centroid
        cx, cy = f2.mean(), f1.mean()
        ax.scatter(cx, cy, c=colors[l1], marker=markers[l1], s=60, zorder=3)
        # Confidence ellipse (95%)
        try:
            cov = np.cov(f2, f1)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            order = eigenvalues.argsort()[::-1]
            eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
            angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
            chi2_val = stats.chi2.ppf(0.95, df=2)
            width, height = 2 * np.sqrt(chi2_val * eigenvalues)
            from matplotlib.patches import Ellipse
            ell = Ellipse(xy=(cx, cy), width=width, height=height,
                          angle=angle, linewidth=1.2,
                          edgecolor=colors[l1], facecolor="none", alpha=0.6)
            ax.add_patch(ell)
        except Exception:
            pass
        if l1 == "L1":
            ax.annotate(phoneme, (cx, cy), fontsize=8, ha="center", va="bottom")

    ax.invert_yaxis()
    ax.set_xlabel("F2 (Lobanov)")
    ax.set_ylabel("F1 (Lobanov)")
    ax.set_title("Vowel Chart – F1 vs F2 (Lobanov normalised)")
    from matplotlib.lines import Line2D
    legend_els = [Line2D([0], [0], marker="o", color="w", markerfacecolor=colors["L1"],
                         markersize=8, label="L1"),
                  Line2D([0], [0], marker="^", color="w", markerfacecolor=colors["L2"],
                         markersize=8, label="L2")]
    ax.legend(handles=legend_els)
    savefig(fig, fig_dir / "5_1_vowel_chart.png")

    # ── 5.1d Box plots F1/F2 per phoneme by L1 status ────────────────────────
    for feat, label in [("F1_lob", "F1"), ("F2_lob", "F2")]:
        fig, ax = plt.subplots(figsize=(14, 5))
        phonemes_present = [p for p in ORAL_VOWELS if p in vowel_df["phoneme_base"].unique()]
        data_l1 = [vowel_df[(vowel_df["phoneme_base"]==p) & (vowel_df["l1_status"]=="L1")][feat].dropna()
                   for p in phonemes_present]
        data_l2 = [vowel_df[(vowel_df["phoneme_base"]==p) & (vowel_df["l1_status"]=="L2")][feat].dropna()
                   for p in phonemes_present]
        x = np.arange(len(phonemes_present))
        bp1 = ax.boxplot(data_l1, positions=x - 0.2, widths=0.3, patch_artist=True,
                         boxprops=dict(facecolor="#BBDEFB"), medianprops=dict(color="navy"))
        bp2 = ax.boxplot(data_l2, positions=x + 0.2, widths=0.3, patch_artist=True,
                         boxprops=dict(facecolor="#FFCDD2"), medianprops=dict(color="darkred"))
        ax.set_xticks(x)
        ax.set_xticklabels(phonemes_present)
        ax.set_xlabel("Phoneme")
        ax.set_ylabel(f"{label} (Lobanov)")
        ax.set_title(f"{label} distribution by phoneme and L1 status")
        ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ["L1", "L2"])
        savefig(fig, fig_dir / f"5_1_{label.lower()}_boxplot.png")

    log.info("  Section 5.1 done.")
    return results


def section5_neural(data: dict, fig_dir: Path) -> dict:
    log.info("Section 5.2: Neural descriptive statistics …")
    results = {}

    for label, npz_key, meta_key in [
        ("XLS-R",   "xlsr_npz",    "xlsr_meta"),
        ("Whisper", "whisper_npz", "whisper_meta"),
    ]:
        npz  = data[npz_key]
        meta = data[meta_key]
        if npz is None:
            log.warning(f"  {label} not available, skipping.")
            continue

        layers = npz["layers"]
        pca_2d = npz["pca_2d"]    # (N, n_layers, 2)
        umap_2d = npz["umap_2d"]  # (N, n_layers, 2)

        for li, layer_idx in enumerate(layers):
            tag = f"{label.lower()}_L{layer_idx}"

            for proj_name, proj_data in [("PCA", pca_2d), ("UMAP", umap_2d)]:
                X2 = proj_data[:, li, :]   # (N, 2)

                # Colour by phoneme
                phoneme_base = meta["phoneme"].apply(
                    lambda p: p if p in ORAL_VOWELS else "other"
                )
                unique_ph = [p for p in ORAL_VOWELS if p in phoneme_base.unique()]
                cmap = plt.cm.get_cmap("tab20", len(unique_ph) + 1)

                fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                for ax, col, col_label in zip(
                    axes,
                    ["phoneme_base", "l1_status", "gender"],
                    ["Phoneme", "L1 Status", "Gender"],
                ):
                    if col == "phoneme_base":
                        vals = phoneme_base
                    else:
                        vals = meta[col]
                    le = LabelEncoder()
                    c = le.fit_transform(vals.fillna("?"))
                    sc = ax.scatter(X2[:, 0], X2[:, 1], c=c, cmap="tab10",
                                   s=4, alpha=0.5)
                    ax.set_title(f"Coloured by {col_label}")
                    ax.set_xlabel(f"{proj_name} 1")
                    ax.set_ylabel(f"{proj_name} 2")

                fig.suptitle(f"{label} Layer {layer_idx} – {proj_name} projection")
                savefig(fig, fig_dir / f"5_2_{tag}_{proj_name.lower()}.png")

            # Between-class variance ratio
            vowel_mask = meta["phoneme"].isin(ORAL_VOWELS)
            X_vowels   = pca_2d[vowel_mask, li, :]
            labels_v   = meta["phoneme"][vowel_mask].values
            if len(np.unique(labels_v)) > 1:
                overall_mean = X_vowels.mean(axis=0)
                between_var = sum(
                    np.sum(labels_v == c) * np.linalg.norm(X_vowels[labels_v == c].mean(0) - overall_mean) ** 2
                    for c in np.unique(labels_v)
                ) / len(X_vowels)
                total_var = X_vowels.var(axis=0).sum()
                bcvr = between_var / total_var if total_var > 0 else 0.0
                results[f"{tag}_bcvr"] = round(float(bcvr), 4)
                log.info(f"  {tag} between-class variance ratio = {bcvr:.4f}")

            # Within/between phoneme cosine similarity
            pca_50 = npz["pca_50d"][:, li, :]
            within_sims, between_sims = [], []
            vowel_phonemes = [p for p in ORAL_VOWELS if (meta["phoneme"] == p).sum() >= 2]
            for ph in vowel_phonemes[:8]:  # limit for speed
                idx = meta["phoneme"][meta["phoneme"] == ph].index
                vecs = pca_50[idx]
                for ia in range(min(10, len(vecs))):
                    for ib in range(ia + 1, min(10, len(vecs))):
                        sim = 1 - cosine_distance(vecs[ia], vecs[ib])
                        within_sims.append(sim)
            for ph_a, ph_b in [("a","i"), ("i","u"), ("e","o"), ("ɛ","ɔ")]:
                if ph_a in vowel_phonemes and ph_b in vowel_phonemes:
                    va = pca_50[meta["phoneme"][meta["phoneme"]==ph_a].index[:5]]
                    vb = pca_50[meta["phoneme"][meta["phoneme"]==ph_b].index[:5]]
                    for a in va:
                        for b in vb:
                            between_sims.append(1 - cosine_distance(a, b))

            if within_sims and between_sims:
                results[f"{tag}_within_cosine_sim"]  = round(float(np.mean(within_sims)), 4)
                results[f"{tag}_between_cosine_sim"] = round(float(np.mean(between_sims)), 4)
                results[f"{tag}_cosine_ratio"] = round(
                    float(np.mean(within_sims) / max(abs(np.mean(between_sims)), 1e-9)), 4
                )

    log.info("  Section 5.2 done.")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Section 6 – Statistical Tests
# ─────────────────────────────────────────────────────────────────────────────

def section6_group_comparisons(df_ac: pd.DataFrame, data: dict,
                                params: dict, fig_dir: Path) -> dict:
    log.info("Section 6.1: Group comparisons …")
    results = {}
    B = params["analysis"]["permutation_B"]

    vowel_df = df_ac[df_ac["phoneme_base"].isin(ORAL_VOWELS)].copy()

    # ── 6.1a L1 vs L2 on acoustic F1/F2 ──────────────────────────────────────
    acoustic_test_rows = []
    for feat in ["F1_lob", "F2_lob"]:
        pvals = []
        phonemes_tested = []
        for phoneme in ORAL_VOWELS:
            grp = vowel_df[vowel_df["phoneme_base"] == phoneme]
            l1  = grp[grp["l1_status"] == "L1"][feat].dropna()
            l2  = grp[grp["l1_status"] == "L2"][feat].dropna()
            if len(l1) < 3 or len(l2) < 3:
                continue
            # Normality check
            _, p_sw_l1 = stats.shapiro(l1[:50])
            _, p_sw_l2 = stats.shapiro(l2[:50])
            # Levene
            _, p_lev = stats.levene(l1, l2)
            if p_sw_l1 > 0.05 and p_sw_l2 > 0.05:
                stat, p = stats.ttest_ind(l1, l2, equal_var=(p_lev > 0.05))
                test_used = "t-test"
            else:
                stat, p = stats.mannwhitneyu(l1, l2, alternative="two-sided")
                test_used = "Mann-Whitney"
            pvals.append(float(p))
            phonemes_tested.append(phoneme)
            acoustic_test_rows.append({
                "feature": feat, "phoneme": phoneme,
                "test": test_used, "stat": round(float(stat), 4),
                "p_raw": round(float(p), 6),
                "n_L1": len(l1), "n_L2": len(l2),
                "mean_L1": round(float(l1.mean()), 4),
                "mean_L2": round(float(l2.mean()), 4),
            })

        # BH correction
        if pvals:
            adj = bh_correction(pvals)
            for i, row in enumerate([r for r in acoustic_test_rows if r["feature"] == feat]):
                row["p_adj_BH"] = round(float(adj[i]), 6)
                row["sig_BH"]   = adj[i] < 0.05

    results["acoustic_group_tests"] = acoustic_test_rows
    log.info(f"  Acoustic tests: {len(acoustic_test_rows)} phoneme×feature pairs tested.")

    # ── 6.1b Neural permutation test ─────────────────────────────────────────
    neural_test_rows = []
    rng = np.random.default_rng(42)

    for label, npz_key, meta_key in [
        ("XLS-R",   "xlsr_npz",    "xlsr_meta"),
        ("Whisper", "whisper_npz", "whisper_meta"),
    ]:
        npz  = data[npz_key]
        meta = data[meta_key]
        if npz is None:
            continue

        layers = npz["layers"]
        pca_50 = npz["pca_50d"]

        for li, layer_idx in enumerate(layers):
            X = pca_50[:, li, :]
            pvals = []
            phonemes_tested = []

            for phoneme in ORAL_VOWELS:
                mask = meta["phoneme"] == phoneme
                if mask.sum() < 6:
                    continue
                X_ph = X[mask]
                l1_mask = meta["l1_status"][mask] == "L1"
                l2_mask = meta["l1_status"][mask] == "L2"
                if l1_mask.sum() < 3 or l2_mask.sum() < 3:
                    continue

                c_l1 = X_ph[l1_mask.values].mean(axis=0)
                c_l2 = X_ph[l2_mask.values].mean(axis=0)
                obs_dist = cosine_distance(c_l1, c_l2)

                # Permutation
                labels_ph = meta["l1_status"][mask].values
                null_dists = []
                for _ in range(min(B, 500)):  # cap for speed
                    perm = rng.permutation(labels_ph)
                    c1 = X_ph[perm == "L1"].mean(axis=0) if (perm == "L1").any() else c_l1
                    c2 = X_ph[perm == "L2"].mean(axis=0) if (perm == "L2").any() else c_l2
                    null_dists.append(cosine_distance(c1, c2))
                p_perm = (np.array(null_dists) >= obs_dist).mean()

                pvals.append(float(p_perm))
                phonemes_tested.append(phoneme)
                neural_test_rows.append({
                    "model": label, "layer": int(layer_idx),
                    "phoneme": phoneme,
                    "obs_cosine_dist": round(float(obs_dist), 6),
                    "p_perm": round(float(p_perm), 6),
                })

            if pvals:
                adj = bh_correction(pvals)
                rows_this = [r for r in neural_test_rows
                             if r["model"] == label and r["layer"] == int(layer_idx)]
                for i, row in enumerate(rows_this):
                    row["p_adj_BH"] = round(float(adj[i]), 6)
                    row["sig_BH"]   = adj[i] < 0.05

    results["neural_group_tests"] = neural_test_rows
    log.info(f"  Neural permutation tests: {len(neural_test_rows)} rows.")
    log.info("  Section 6.1 done.")
    return results


def section6_distances(df_ac: pd.DataFrame, data: dict,
                        params: dict, fig_dir: Path) -> dict:
    log.info("Section 6.2: Inter-phoneme distances …")
    results = {}
    B = params["analysis"]["bootstrap_B"]

    vowel_df = df_ac[df_ac["phoneme_base"].isin(ORAL_VOWELS)].copy()
    present_vowels = [p for p in ORAL_VOWELS if (vowel_df["phoneme_base"] == p).sum() >= 3]

    # ── Acoustic distance matrices (Euclidean + Mahalanobis) ──────────────────
    centroids_ac = np.array([
        vowel_df[vowel_df["phoneme_base"] == p][["F1_lob", "F2_lob"]].mean().values
        for p in present_vowels
    ])

    # Pooled within-class covariance
    pool_cov = np.zeros((2, 2))
    n_total  = 0
    for p in present_vowels:
        sub = vowel_df[vowel_df["phoneme_base"] == p][["F1_lob", "F2_lob"]].dropna()
        if len(sub) > 2:
            pool_cov += np.cov(sub.values.T) * (len(sub) - 1)
            n_total  += len(sub) - 1
    if n_total > 0:
        pool_cov /= n_total

    n = len(present_vowels)
    D_euc  = np.zeros((n, n))
    D_mah  = np.zeros((n, n))
    try:
        pool_cov_inv = np.linalg.inv(pool_cov + np.eye(2) * 1e-6)
    except Exception:
        pool_cov_inv = np.eye(2)

    for i in range(n):
        for j in range(i + 1, n):
            diff = centroids_ac[i] - centroids_ac[j]
            D_euc[i, j] = D_euc[j, i] = float(np.linalg.norm(diff))
            D_mah[i, j] = D_mah[j, i] = float(np.sqrt(diff @ pool_cov_inv @ diff))

    results["acoustic_distance_matrix"] = {
        "phonemes": present_vowels,
        "euclidean": D_euc.tolist(),
        "mahalanobis": D_mah.tolist(),
    }

    # ── Neural distance matrices ──────────────────────────────────────────────
    mantel_results = {}

    for label, npz_key, meta_key in [
        ("XLS-R",   "xlsr_npz",    "xlsr_meta"),
        ("Whisper", "whisper_npz", "whisper_meta"),
    ]:
        npz  = data[npz_key]
        meta = data[meta_key]
        if npz is None:
            continue

        layers = npz["layers"]
        pca_50 = npz["pca_50d"]
        li     = len(layers) // 2   # use middle layer for distance comparisons

        centroids_neural = []
        neural_present   = []
        for p in present_vowels:
            idx = meta["phoneme"][meta["phoneme"] == p].index
            if len(idx) < 2:
                continue
            centroids_neural.append(pca_50[idx, li, :].mean(axis=0))
            neural_present.append(p)

        if len(neural_present) < 3:
            continue

        centroids_neural = np.array(centroids_neural)
        D_neural = cdist(centroids_neural, centroids_neural, metric="cosine")

        # Align to acoustic matrix
        common = [p for p in present_vowels if p in neural_present]
        ac_idx = [present_vowels.index(p) for p in common]
        ne_idx = [neural_present.index(p)  for p in common]
        D_ac_sub = D_euc[np.ix_(ac_idx, ac_idx)]
        D_ne_sub = D_neural[np.ix_(ne_idx, ne_idx)]

        r, p = mantel_test(D_ac_sub, D_ne_sub, n_perm=499)
        mantel_results[f"acoustic_vs_{label.lower()}"] = {"r": round(r, 4), "p": round(p, 4)}
        log.info(f"  Mantel: acoustic vs {label} r={r:.4f} p={p:.4f}")

    results["mantel_tests"] = mantel_results

    # ── Nearest-centroid LOSO classifier ─────────────────────────────────────
    log.info("  Running LOSO nearest-centroid classifier …")
    classifier_results = {}
    speakers = vowel_df["speaker"].unique()

    for feat_label, feat_cols in [("acoustic", ["F1_lob", "F2_lob"])]:
        all_true, all_pred = [], []
        for test_spk in speakers:
            train = vowel_df[vowel_df["speaker"] != test_spk]
            test  = vowel_df[vowel_df["speaker"] == test_spk]
            centroids = {
                p: train[train["phoneme_base"] == p][feat_cols].mean().values
                for p in present_vowels
            }
            for _, row in test[test["phoneme_base"].isin(present_vowels)].iterrows():
                x = row[feat_cols].values.astype(float)
                if np.isnan(x).any():
                    continue
                dists = {p: np.linalg.norm(x - c) for p, c in centroids.items()
                         if not np.isnan(c).any()}
                if dists:
                    pred = min(dists, key=dists.get)
                    all_true.append(row["phoneme_base"])
                    all_pred.append(pred)

        if all_true:
            acc = accuracy_score(all_true, all_pred)
            f1  = f1_score(all_true, all_pred, average="macro", zero_division=0)
            classifier_results[feat_label] = {
                "accuracy": round(float(acc), 4),
                "f1_macro": round(float(f1), 4),
                "n_test":   len(all_true),
            }
            log.info(f"  LOSO {feat_label}: acc={acc:.4f} F1={f1:.4f}")

            # Confusion matrix
            le = LabelEncoder()
            le.fit(present_vowels)
            cm = confusion_matrix(all_true, all_pred, labels=present_vowels)
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xticks(range(len(present_vowels)))
            ax.set_yticks(range(len(present_vowels)))
            ax.set_xticklabels(present_vowels, rotation=45, ha="right")
            ax.set_yticklabels(present_vowels)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(f"Confusion Matrix – {feat_label} (LOSO)")
            plt.colorbar(im, ax=ax)
            savefig(fig, fig_dir / f"6_2_confusion_{feat_label}.png")

    results["loso_classifier"] = classifier_results
    log.info("  Section 6.2 done.")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Section 7 – Linear Mixed-Effects Models
# ─────────────────────────────────────────────────────────────────────────────

def section7_lme(df_ac: pd.DataFrame, params: dict) -> dict:
    log.info("Section 7: Linear Mixed-Effects Models …")
    results = {}

    try:
        import statsmodels.formula.api as smf
    except ImportError:
        log.warning("  statsmodels not installed – skipping LME. pip install statsmodels")
        return results

    vowel_df = df_ac[df_ac["phoneme_base"].isin(ORAL_VOWELS)].copy()
    vowel_df["is_L2"]   = (vowel_df["l1_status"] == "L2").astype(float)
    vowel_df["is_male"] = (vowel_df["gender"] == "M").astype(float)

    lme_results = {}

    for response in ["F1_lob", "F2_lob"]:
        sub = vowel_df[["speaker", response, "is_L2", "is_male", "phoneme_base"]].dropna()
        if len(sub) < 20:
            continue

        models = {}

        try:
            # Null model (for ICC)
            null = smf.mixedlm(f"{response} ~ 1", sub, groups=sub["speaker"]).fit(
                method="lbfgs", reml=True)
            icc_num = float(null.cov_re.iloc[0, 0])
            icc_den = icc_num + float(null.scale)
            icc = icc_num / icc_den if icc_den > 0 else float("nan")
            models["null_ICC"] = round(icc, 4)
            log.info(f"  {response} ICC = {icc:.4f}")

            # Main effects
            main = smf.mixedlm(
                f"{response} ~ is_L2 + is_male", sub, groups=sub["speaker"]
            ).fit(method="lbfgs", reml=False)
            models["main_effects"] = {
                "AIC":   round(float(main.aic), 2),
                "BIC":   round(float(main.bic), 2),
                "coef_L2":   round(float(main.params.get("is_L2", float("nan"))), 4),
                "pval_L2":   round(float(main.pvalues.get("is_L2", float("nan"))), 6),
                "coef_male": round(float(main.params.get("is_male", float("nan"))), 4),
                "pval_male": round(float(main.pvalues.get("is_male", float("nan"))), 6),
            }

            # Full model with interaction
            full = smf.mixedlm(
                f"{response} ~ is_L2 * is_male", sub, groups=sub["speaker"]
            ).fit(method="lbfgs", reml=False)
            models["full_model"] = {
                "AIC": round(float(full.aic), 2),
                "BIC": round(float(full.bic), 2),
                "coef_interaction": round(
                    float(full.params.get("is_L2:is_male", float("nan"))), 4),
                "pval_interaction": round(
                    float(full.pvalues.get("is_L2:is_male", float("nan"))), 6),
            }

            # LRT: main vs full
            lrt_stat = -2 * (main.llf - full.llf)
            lrt_p    = float(stats.chi2.sf(lrt_stat, df=1))
            models["LRT_main_vs_full"] = {
                "stat": round(float(lrt_stat), 4),
                "p":    round(lrt_p, 6),
            }

            # Marginal R² (variance of fixed effects / total variance)
            fitted_var = np.var(full.fittedvalues)
            total_var  = np.var(sub[response])
            models["marginal_R2"] = round(
                float(fitted_var / total_var) if total_var > 0 else float("nan"), 4)

        except Exception as e:
            log.warning(f"  LME for {response} failed: {e}")

        lme_results[response] = models

    results["lme"] = lme_results
    log.info("  Section 7 done.")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Section 8 – Confidence Intervals and ROPE
# ─────────────────────────────────────────────────────────────────────────────

def section8_rope(df_ac: pd.DataFrame, data: dict, params: dict, fig_dir: Path) -> dict:
    log.info("Section 8: Confidence Intervals and ROPE …")
    results = {}
    B        = params["analysis"]["bootstrap_B"]
    rope_hz  = params["analysis"]["acoustic_rope_hz"]

    vowel_df = df_ac[df_ac["phoneme_base"].isin(ORAL_VOWELS)].copy()
    speakers = vowel_df["speaker"].unique()

    # ── 8.1 Bootstrap CIs on acoustic L1/L2 contrast ─────────────────────────
    ac_contrast_rows = []
    for feat, feat_label in [("F1_lob", "F1"), ("F2_lob", "F2")]:
        for phoneme in ORAL_VOWELS:
            grp = vowel_df[vowel_df["phoneme_base"] == phoneme]
            l1  = grp[grp["l1_status"] == "L1"][feat].dropna().values
            l2  = grp[grp["l1_status"] == "L2"][feat].dropna().values
            if len(l1) < 3 or len(l2) < 3:
                continue
            point_est = float(l1.mean() - l2.mean())
            # Speaker-level bootstrap: resample speakers
            rng = np.random.default_rng(42)
            boot_diffs = []
            for _ in range(B):
                spk_boot = rng.choice(speakers, size=len(speakers), replace=True)
                l1_b = vowel_df[
                    vowel_df["speaker"].isin(spk_boot) &
                    (vowel_df["phoneme_base"] == phoneme) &
                    (vowel_df["l1_status"] == "L1")
                ][feat].dropna().values
                l2_b = vowel_df[
                    vowel_df["speaker"].isin(spk_boot) &
                    (vowel_df["phoneme_base"] == phoneme) &
                    (vowel_df["l1_status"] == "L2")
                ][feat].dropna().values
                if len(l1_b) > 0 and len(l2_b) > 0:
                    boot_diffs.append(float(l1_b.mean() - l2_b.mean()))
            if len(boot_diffs) < 10:
                continue
            ci_lo = float(np.percentile(boot_diffs, 2.5))
            ci_hi = float(np.percentile(boot_diffs, 97.5))
            # ROPE classification (acoustic: normalised units, not Hz)
            # Use ±0.1 Lobanov units as approximate ROPE (≈20 Hz at typical F1)
            rope_lo, rope_hi = -0.1, 0.1
            if ci_lo > rope_hi or ci_hi < rope_lo:
                rope_class = "non-equivalent"
            elif ci_lo >= rope_lo and ci_hi <= rope_hi:
                rope_class = "equivalent"
            else:
                rope_class = "indeterminate"
            ac_contrast_rows.append({
                "feature": feat_label, "phoneme": phoneme,
                "point_est": round(point_est, 4),
                "ci_lo": round(ci_lo, 4), "ci_hi": round(ci_hi, 4),
                "rope_class": rope_class,
            })

    results["acoustic_rope"] = ac_contrast_rows

    # ── Forest plot – acoustic ────────────────────────────────────────────────
    for feat_label in ["F1", "F2"]:
        rows = [r for r in ac_contrast_rows if r["feature"] == feat_label]
        if not rows:
            continue
        rows_sorted = sorted(rows, key=lambda r: r["point_est"])
        fig, ax = plt.subplots(figsize=(8, max(4, len(rows_sorted) * 0.4 + 1)))
        colors_rope = {"equivalent": "green", "non-equivalent": "red",
                       "indeterminate": "orange"}
        for i, row in enumerate(rows_sorted):
            c = colors_rope[row["rope_class"]]
            ax.plot([row["ci_lo"], row["ci_hi"]], [i, i], color=c, linewidth=2)
            ax.scatter([row["point_est"]], [i], color=c, s=40, zorder=3)
            ax.axvline(0, color="black", linewidth=0.5)
            ax.axvspan(-0.1, 0.1, alpha=0.08, color="gray", label="ROPE" if i == 0 else "")
        ax.set_yticks(range(len(rows_sorted)))
        ax.set_yticklabels([r["phoneme"] for r in rows_sorted])
        ax.set_xlabel(f"{feat_label} L1−L2 contrast (Lobanov units)")
        ax.set_title(f"Forest plot: {feat_label} L1/L2 contrast with 95% CI")
        ax.legend()
        savefig(fig, fig_dir / f"8_forest_{feat_label.lower()}.png")

    # ── 8.2 Neural ROPE ──────────────────────────────────────────────────────
    neural_rope_rows = []
    for label, npz_key, meta_key in [
        ("XLS-R",   "xlsr_npz",    "xlsr_meta"),
        ("Whisper", "whisper_npz", "whisper_meta"),
    ]:
        npz  = data[npz_key]
        meta = data[meta_key]
        if npz is None:
            continue

        layers = npz["layers"]
        pca_50 = npz["pca_50d"]
        li     = 0  # use first layer for ROPE analysis

        # Compute neural ROPE: mean intra-speaker cosine distance
        intra_dists = []
        for spk in meta["speaker"].unique()[:5]:
            for ph in ORAL_VOWELS[:4]:
                idx = meta[(meta["speaker"] == spk) & (meta["phoneme"] == ph)].index
                if len(idx) < 2:
                    continue
                vecs = pca_50[idx, li, :]
                for ia in range(min(5, len(vecs))):
                    for ib in range(ia + 1, min(5, len(vecs))):
                        intra_dists.append(cosine_distance(vecs[ia], vecs[ib]))
        delta0 = float(np.mean(intra_dists)) if intra_dists else 0.05

        rng = np.random.default_rng(42)
        for phoneme in ORAL_VOWELS:
            mask = meta["phoneme"] == phoneme
            if mask.sum() < 6:
                continue
            X_ph    = pca_50[mask, li, :]
            l1_mask = (meta["l1_status"][mask] == "L1").values
            l2_mask = (meta["l1_status"][mask] == "L2").values
            if l1_mask.sum() < 3 or l2_mask.sum() < 3:
                continue
            c_l1 = X_ph[l1_mask].mean(axis=0)
            c_l2 = X_ph[l2_mask].mean(axis=0)
            obs   = cosine_distance(c_l1, c_l2)

            # Speaker-level bootstrap CI
            boot_dists = []
            for _ in range(B):
                spk_boot = rng.choice(speakers, size=len(speakers), replace=True)
                idx_b = meta[meta["speaker"].isin(spk_boot) &
                             (meta["phoneme"] == phoneme)].index
                if len(idx_b) < 6:
                    continue
                l1b = pca_50[idx_b[meta["l1_status"][idx_b] == "L1"], li, :]
                l2b = pca_50[idx_b[meta["l1_status"][idx_b] == "L2"], li, :]
                if len(l1b) > 0 and len(l2b) > 0:
                    boot_dists.append(cosine_distance(l1b.mean(0), l2b.mean(0)))
            if len(boot_dists) < 10:
                continue
            ci_lo = float(np.percentile(boot_dists, 2.5))
            ci_hi = float(np.percentile(boot_dists, 97.5))
            if ci_lo > delta0:
                rope_class = "non-equivalent"
            elif ci_hi <= delta0:
                rope_class = "equivalent"
            else:
                rope_class = "indeterminate"
            neural_rope_rows.append({
                "model": label, "layer": int(layers[li]), "phoneme": phoneme,
                "point_est": round(float(obs), 6),
                "ci_lo": round(ci_lo, 6), "ci_hi": round(ci_hi, 6),
                "delta0": round(delta0, 6),
                "rope_class": rope_class,
            })

    results["neural_rope"] = neural_rope_rows
    log.info(f"  ROPE: {len(ac_contrast_rows)} acoustic, {len(neural_rope_rows)} neural rows.")
    log.info("  Section 8 done.")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Section 9 – Hierarchical Clustering
# ─────────────────────────────────────────────────────────────────────────────

def section9_clustering(df_ac: pd.DataFrame, data: dict,
                         params: dict, fig_dir: Path) -> dict:
    log.info("Section 9: Hierarchical Clustering …")
    results = {}
    link_method = params["analysis"]["linkage"]

    vowel_df = df_ac[df_ac["phoneme_base"].isin(ORAL_VOWELS)].copy()

    # ── Ground-truth partition vectors ────────────────────────────────────────
    def front_back(p): return 0 if p in FRONT_VOWELS else 1
    def high_low(p):
        if p in HIGH_VOWELS: return 0
        if p in MID_VOWELS:  return 1
        return 2

    ari_results = {}

    # ── 9.1 Acoustic clustering of French oral vowels ─────────────────────────
    present = [p for p in ORAL_VOWELS
               if (vowel_df["phoneme_base"] == p).sum() >= 3]
    centroids_ac = np.array([
        vowel_df[vowel_df["phoneme_base"] == p][["F1_lob", "F2_lob"]].mean().values
        for p in present
    ])

    if len(present) >= 3:
        Z_ac = linkage(centroids_ac, method=link_method, metric="euclidean")

        fig, ax = plt.subplots(figsize=(10, 5))
        dendrogram(Z_ac, labels=present, ax=ax, orientation="top")
        ax.set_title("Acoustic clustering of French oral vowels")
        ax.set_ylabel("Distance")
        savefig(fig, fig_dir / "9_1_acoustic_dendrogram.png")

        # ARI against front/back
        k = 2
        clusters = fcluster(Z_ac, k, criterion="maxclust")
        gt_fb = [front_back(p) for p in present]
        gt_hl = [high_low(p) for p in present]
        ari_fb = adjusted_rand_score(gt_fb, clusters)
        ari_hl = adjusted_rand_score(gt_hl, fcluster(Z_ac, 3, criterion="maxclust"))
        ari_results["acoustic_front_back"] = round(float(ari_fb), 4)
        ari_results["acoustic_high_low"]   = round(float(ari_hl), 4)
        log.info(f"  Acoustic ARI front/back={ari_fb:.4f}, high/low={ari_hl:.4f}")

    # ── 9.1 Neural clustering ─────────────────────────────────────────────────
    for label, npz_key, meta_key in [
        ("XLS-R",   "xlsr_npz",    "xlsr_meta"),
        ("Whisper", "whisper_npz", "whisper_meta"),
    ]:
        npz  = data[npz_key]
        meta = data[meta_key]
        if npz is None:
            continue

        layers = npz["layers"]
        pca_50 = npz["pca_50d"]

        for li, layer_idx in enumerate(layers):
            tag = f"{label.lower()}_L{layer_idx}"
            neural_present = [p for p in ORAL_VOWELS
                              if (meta["phoneme"] == p).sum() >= 3]
            if len(neural_present) < 3:
                continue

            centroids_ne = np.array([
                pca_50[meta["phoneme"][meta["phoneme"] == p].index, li, :].mean(axis=0)
                for p in neural_present
            ])
            # Use cosine distance → convert to condensed form
            D = cdist(centroids_ne, centroids_ne, metric="cosine")
            D = (D + D.T) / 2
            np.fill_diagonal(D, 0)
            condensed = squareform(D)
            Z_ne = linkage(condensed, method=link_method)

            fig, ax = plt.subplots(figsize=(10, 5))
            dendrogram(Z_ne, labels=neural_present, ax=ax)
            ax.set_title(f"{label} L{layer_idx} – vowel clustering")
            savefig(fig, fig_dir / f"9_1_{tag}_dendrogram.png")

            gt_fb = [front_back(p) for p in neural_present]
            gt_hl = [high_low(p)   for p in neural_present]
            ari_fb = adjusted_rand_score(gt_fb, fcluster(Z_ne, 2, criterion="maxclust"))
            ari_hl = adjusted_rand_score(gt_hl, fcluster(Z_ne, 3, criterion="maxclust"))
            ari_results[f"{tag}_front_back"] = round(float(ari_fb), 4)
            ari_results[f"{tag}_high_low"]   = round(float(ari_hl), 4)
            log.info(f"  {tag} ARI front/back={ari_fb:.4f}, high/low={ari_hl:.4f}")

    results["ari_vowel_clustering"] = ari_results

    # ── 9.3 Speaker clustering ────────────────────────────────────────────────
    log.info("  Speaker clustering …")
    speaker_ari = {}

    # Acoustic: per-speaker per-phoneme mean F1/F2
    spk_list = vowel_df["speaker"].unique()
    spk_feat_ac = []
    for spk in spk_list:
        row = []
        for p in present:
            sub = vowel_df[(vowel_df["speaker"] == spk) &
                           (vowel_df["phoneme_base"] == p)][["F1_lob", "F2_lob"]].mean()
            row.extend(sub.values.tolist())
        spk_feat_ac.append(row)
    spk_feat_ac = np.nan_to_num(np.array(spk_feat_ac))

    if len(spk_list) >= 4:
        Z_spk = linkage(spk_feat_ac, method=link_method, metric="euclidean")
        gt_l1 = LabelEncoder().fit_transform(
            [vowel_df[vowel_df["speaker"] == s]["l1_status"].iloc[0] for s in spk_list])
        gt_gen = LabelEncoder().fit_transform(
            [vowel_df[vowel_df["speaker"] == s]["gender"].iloc[0] for s in spk_list])
        ari_l1  = adjusted_rand_score(gt_l1,  fcluster(Z_spk, 2, criterion="maxclust"))
        ari_gen = adjusted_rand_score(gt_gen, fcluster(Z_spk, 2, criterion="maxclust"))
        speaker_ari["acoustic_L1_status"] = round(float(ari_l1), 4)
        speaker_ari["acoustic_gender"]    = round(float(ari_gen), 4)
        log.info(f"  Speaker ARI (acoustic): L1={ari_l1:.4f} gender={ari_gen:.4f}")

        fig, ax = plt.subplots(figsize=(10, 5))
        dendrogram(Z_spk, labels=list(spk_list), ax=ax)
        ax.set_title("Speaker clustering (acoustic features)")
        savefig(fig, fig_dir / "9_3_speaker_dendrogram_acoustic.png")

    results["ari_speaker_clustering"] = speaker_ari

    # ── 9.4 Silhouette analysis ───────────────────────────────────────────────
    if len(present) >= 4:
        sil_scores = {}
        for k in range(2, min(params["analysis"]["k_max"] + 1, len(present))):
            clusters_k = fcluster(Z_ac, k, criterion="maxclust")
            try:
                sil = silhouette_score(centroids_ac, clusters_k)
                sil_scores[k] = round(float(sil), 4)
            except Exception:
                pass
        results["silhouette_scores"] = sil_scores
        if sil_scores:
            best_k = max(sil_scores, key=sil_scores.get)
            log.info(f"  Best k by silhouette: k={best_k} (score={sil_scores[best_k]:.4f})")

        # Silhouette plot
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(list(sil_scores.keys()), list(sil_scores.values()), "o-")
        ax.set_xlabel("k")
        ax.set_ylabel("Silhouette coefficient")
        ax.set_title("Silhouette scores for acoustic vowel clustering")
        savefig(fig, fig_dir / "9_4_silhouette.png")

    log.info("  Section 9 done.")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    params   = load_params()
    fig_dir  = Path(params["analysis"]["figures_dir"])
    out_dir  = Path(params["analysis"]["output_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(params)

    if data["acoustic"] is None:
        log.error("Acoustic features not found. Run extract_acoustics + normalise first.")
        return

    df_ac = data["acoustic"]

    # Add phoneme_base column if missing
    if "phoneme_base" not in df_ac.columns:
        PHONEME_BASE_MAP = {
            "aː": "a", "u:": "u", "ɑ̃ː": "ɑ̃",
            "a̰": "a", "i̥": "i", "y̥": "y",
            "ə̰": "ə", "ɛ̰": "ɛ", "ø̰": "ø", "ɑ̰̃": "ɑ̃",
        }
        df_ac["phoneme_base"] = df_ac["phoneme"].apply(
            lambda p: PHONEME_BASE_MAP.get(p, p))

    all_results = {}

    log.info("═" * 60)
    log.info("Running Section 5: Descriptive Statistics")
    r5a = section5_acoustic(df_ac, fig_dir)
    all_results.update(r5a)

    if data["xlsr_npz"] is not None or data["whisper_npz"] is not None:
        r5n = section5_neural(data, fig_dir)
        all_results.update(r5n)
    else:
        log.warning("Neural data not available — Sections 5.2, 6.1b, 7.2, 8.2, 9.1b skipped.")

    log.info("═" * 60)
    log.info("Running Section 6: Statistical Tests")
    r6a = section6_group_comparisons(df_ac, data, params, fig_dir)
    all_results.update(r6a)
    r6b = section6_distances(df_ac, data, params, fig_dir)
    all_results.update(r6b)

    log.info("═" * 60)
    log.info("Running Section 7: LME Models")
    r7 = section7_lme(df_ac, params)
    all_results.update(r7)

    log.info("═" * 60)
    log.info("Running Section 8: CIs and ROPE")
    r8 = section8_rope(df_ac, data, params, fig_dir)
    all_results.update(r8)

    log.info("═" * 60)
    log.info("Running Section 9: Hierarchical Clustering")
    r9 = section9_clustering(df_ac, data, params, fig_dir)
    all_results.update(r9)

    # ── Save summary ──────────────────────────────────────────────────────────
    def make_serialisable(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: make_serialisable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serialisable(v) for v in obj]
        return obj

    summary_path = out_dir / "stats_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(make_serialisable(all_results), f, indent=2, ensure_ascii=False)
    log.info(f"Saved summary to {summary_path}")
    log.info("Analysis complete.")


if __name__ == "__main__":
    main()
