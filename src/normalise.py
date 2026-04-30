"""
Stage 5 – normalise.py
───────────────────────
Applies:
  1. Lobanov normalisation to F1/F2 acoustic features
     (z-score per speaker per formant, computed on vowel tokens only)
  2. PCA to neural representations (XLS-R and Whisper)
     - d=2  for visualisation
     - d=50 for clustering / LME (capped at n_components if data is smaller)
  3. UMAP to neural representations (d=2, for scatter plots)

Also adds a canonical `phoneme_base` column that maps diacritic variants
to their base IPA symbol (e.g. ɑ̃ː → ɑ̃, aː → a, ə̰ → ə) so downstream
analyses can group variants together.

Outputs
───────
data/normalised/features_acoustic_norm.csv
data/normalised/features_xlsr_pca.npz
data/normalised/features_whisper_pca.npz

Each .npz contains:
    pca_2d       – (N, n_layers, 2)   PCA to 2 components
    pca_50d      – (N, n_layers, 50)  PCA to 50 components (or fewer)
    umap_2d      – (N, n_layers, 2)   UMAP to 2 components
    meta_json    – JSON string copied from source .npz
    layers       – layer index array copied from source .npz
    pca_explained_variance_ratio – (n_layers, 50) explained variance per PC

Usage (called by DVC):
    python src/normalise.py
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── vowel inventory ───────────────────────────────────────────────────────────
# All tokens whose phoneme label starts with one of these base characters
# are treated as vowels for Lobanov normalisation.
VOWEL_BASES = {
    "a", "e", "i", "o", "u", "y",
    "ø", "œ", "ɛ", "ɔ", "ə", "ɑ",
}

# Map from full phoneme label → canonical base used in analyses
# Diacritic variants are collapsed to their base for grouping.
# Add entries here if new variants appear.
PHONEME_BASE_MAP = {
    # length marks
    "aː":  "a",   "u:":  "u",   "ɑ̃ː": "ɑ̃",
    # creaky voice
    "a̰":  "a",   "i̥":  "i",   "y̥":  "y",
    "ə̰":  "ə",   "ɛ̰":  "ɛ",   "ø̰":  "ø",
    "ɑ̰̃": "ɑ̃",
}


def get_base(phoneme: str) -> str:
    """Return canonical base phoneme, collapsing diacritic variants."""
    if phoneme in PHONEME_BASE_MAP:
        return PHONEME_BASE_MAP[phoneme]
    return phoneme


def is_vowel(phoneme: str) -> bool:
    base = get_base(phoneme)
    return base[0] in VOWEL_BASES if base else False


# ── params ────────────────────────────────────────────────────────────────────

def load_params(path="params.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


# ── Lobanov normalisation ─────────────────────────────────────────────────────

def lobanov_normalise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Lobanov (1971) normalisation to F1 and F2.

    For each speaker s and formant j:
        F*_j,s = (F_j,s - mean_j,s) / sd_j,s

    where mean and sd are computed over ALL vowel tokens of speaker s
    (oral + nasal, including diacritic variants — consonants excluded).

    Returns a copy of df with columns F1_lob, F2_lob added.
    F1_lob/F2_lob are NaN for non-vowel tokens (formants were not extracted).
    """
    df = df.copy()
    df["phoneme_base"] = df["phoneme"].apply(get_base)
    df["F1_lob"] = float("nan")
    df["F2_lob"] = float("nan")

    vowel_mask = df["phoneme"].apply(is_vowel)

    for speaker, grp in df[vowel_mask].groupby("speaker"):
        for col, lob_col in [("F1", "F1_lob"), ("F2", "F2_lob")]:
            vals = grp[col].dropna()
            if len(vals) < 2:
                log.warning(f"Speaker {speaker}: too few vowels for Lobanov on {col}.")
                continue
            mu = vals.mean()
            sd = vals.std(ddof=1)
            if sd == 0:
                log.warning(f"Speaker {speaker}: zero SD for {col}, skipping.")
                continue
            idx = grp.index
            df.loc[idx, lob_col] = (df.loc[idx, col] - mu) / sd

    n_normalised = df["F1_lob"].notna().sum()
    log.info(f"Lobanov normalisation: {n_normalised} vowel tokens normalised.")
    return df


# ── neural normalisation ──────────────────────────────────────────────────────

def normalise_neural(npz_path: Path, params: dict, label: str) -> dict:
    """
    Load a (N, n_layers, D) embeddings array and apply PCA + UMAP per layer.

    Returns a dict of arrays ready for np.savez_compressed.
    """
    np_params  = params["normalise"]
    n_pca_viz  = np_params["pca_components_viz"]   # 2
    n_pca_stat = np_params["pca_components_stats"]  # 50
    n_umap     = np_params["umap_components"]        # 2

    log.info(f"Loading {npz_path} …")
    archive    = np.load(npz_path, allow_pickle=True)
    embeddings = archive["embeddings"].astype(np.float32)  # (N, n_layers, D)
    layers     = archive["layers"]
    meta_json  = str(archive["meta_json"])

    N, n_layers, D = embeddings.shape
    log.info(f"  {label}: shape={embeddings.shape}")

    n_pca_stat = min(n_pca_stat, D, N - 1)  # cap at data dimensions

    pca_2d_all   = np.zeros((N, n_layers, n_pca_viz),  dtype=np.float32)
    pca_50d_all  = np.zeros((N, n_layers, n_pca_stat), dtype=np.float32)
    umap_2d_all  = np.zeros((N, n_layers, n_umap),     dtype=np.float32)
    evr_all      = np.zeros((n_layers, n_pca_stat),     dtype=np.float32)

    for li in range(n_layers):
        layer_label = f"{label} layer {layers[li]}"
        X = embeddings[:, li, :]  # (N, D)

        # Standardise before PCA (zero mean, unit variance per dimension)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # PCA – visualisation (2D)
        pca2 = PCA(n_components=n_pca_viz, random_state=42)
        pca_2d_all[:, li, :] = pca2.fit_transform(X_scaled).astype(np.float32)
        log.info(f"  {layer_label}: PCA-2D explained variance = "
                 f"{pca2.explained_variance_ratio_.sum():.3f}")

        # PCA – statistics (50D)
        pca50 = PCA(n_components=n_pca_stat, random_state=42)
        pca_50d_all[:, li, :] = pca50.fit_transform(X_scaled).astype(np.float32)
        evr_all[li, :] = pca50.explained_variance_ratio_.astype(np.float32)
        log.info(f"  {layer_label}: PCA-50D cumulative variance = "
                 f"{pca50.explained_variance_ratio_.cumsum()[-1]:.3f}")

        # UMAP
        try:
            import umap
            reducer = umap.UMAP(
                n_components=n_umap,
                n_neighbors=np_params["umap_n_neighbors"],
                min_dist=np_params["umap_min_dist"],
                random_state=42,
                verbose=False,
            )
            umap_2d_all[:, li, :] = reducer.fit_transform(X_scaled).astype(np.float32)
            log.info(f"  {layer_label}: UMAP done.")
        except ImportError:
            log.warning("  umap-learn not installed — UMAP skipped. "
                        "Install with: pip install umap-learn")
            umap_2d_all[:, li, :] = pca_2d_all[:, li, :]  # fallback to PCA-2D

    return {
        "pca_2d":                       pca_2d_all,
        "pca_50d":                       pca_50d_all,
        "umap_2d":                       umap_2d_all,
        "pca_explained_variance_ratio":  evr_all,
        "layers":                        layers,
        "meta_json":                     np.array(meta_json),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    params  = load_params()
    np_     = params["normalise"]
    out_dir = Path(np_["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Acoustic normalisation ─────────────────────────────────────────────
    acoustic_path = Path(params["acoustics"]["output_dir"]) / "features_acoustic.csv"
    log.info(f"Loading acoustic features from {acoustic_path} …")
    df = pd.read_csv(acoustic_path, encoding="utf-8")
    log.info(f"  {len(df)} tokens loaded.")

    df_norm = lobanov_normalise(df)

    out_csv = Path(np_["lobanov_output"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_norm.to_csv(out_csv, index=False, encoding="utf-8")
    log.info(f"Wrote normalised acoustics to {out_csv}")

    # ── 2. Neural normalisation – XLS-R ──────────────────────────────────────
    xlsr_path = Path(params["xlsr"]["output_dir"]) / "features_xlsr.npz"
    if xlsr_path.exists():
        result = normalise_neural(xlsr_path, params, "XLS-R")
        out_xlsr = out_dir / "features_xlsr_pca.npz"
        np.savez_compressed(out_xlsr, **result)
        log.info(f"Wrote XLS-R normalised to {out_xlsr}")
    else:
        log.warning(f"{xlsr_path} not found — skipping XLS-R normalisation. "
                    "Run extract_neural_xlsr.py on Kaggle first.")

    # ── 3. Neural normalisation – Whisper ─────────────────────────────────────
    whisper_path = Path(params["whisper"]["output_dir"]) / "features_whisper.npz"
    if whisper_path.exists():
        result = normalise_neural(whisper_path, params, "Whisper")
        out_whisper = out_dir / "features_whisper_pca.npz"
        np.savez_compressed(out_whisper, **result)
        log.info(f"Wrote Whisper normalised to {out_whisper}")
    else:
        log.warning(f"{whisper_path} not found — skipping Whisper normalisation. "
                    "Run extract_neural_whisper.py on Kaggle first.")

    log.info("Normalisation complete.")


if __name__ == "__main__":
    main()
