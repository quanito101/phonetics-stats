"""
Stage 3 – extract_neural_xlsr.py
──────────────────────────────────
Extracts XLS-R (facebook/wav2vec2-large-xlsr-53) hidden states
for every phoneme token in data/processed/phonemes.csv.

For each token and each requested layer, the hidden states at time
steps overlapping the phoneme interval are mean-pooled into a single
vector (Equation 1 of the project brief).

Output
──────
data/neural/xlsr/features_xlsr.npz  containing:
    embeddings   – (N, n_layers, D) float32
    meta_json    – JSON string with token metadata
    layers       – 1-D int array of layer indices extracted
    info_json    – JSON string with model/extraction details

data/neural/xlsr/extraction_stats.json

Adapted from numerical-stability/src/extract_embeddings.py:
  - word-level → phoneme-level segmentation
  - single layer → multiple layers (stored as 3-D array)
  - XLS-R specific frame-rate calculation

Usage:
    python src/extract_neural_xlsr.py          # local (CPU, slow)
    python src/extract_neural_xlsr.py --gpu    # Kaggle / GPU machine
"""

import argparse
import csv
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# XLS-R CNN feature encoder output frame rate (≈ 50 Hz, stride = 320 samples at 16 kHz)
XLSR_FRAME_RATE = 50.0  # frames per second
XLSR_MIN_SAMPLES = 400  # minimum audio samples for a valid forward pass


# ── helpers ───────────────────────────────────────────────────────────────────

def load_params(path="params.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def load_manifest(processed_dir: Path) -> list[dict]:
    rows = []
    with open(processed_dir / "phonemes.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["onset"]  = float(row["onset"])
            row["offset"] = float(row["offset"])
            rows.append(row)
    return rows


# ── audio loading (reused from numerical-stability) ───────────────────────────

def load_audio_segment(wav_path: str, onset: float, offset: float, target_sr: int) -> np.ndarray:
    """
    Load [onset, offset] from wav_path, resample to target_sr if needed.
    Returns 1-D float32 array. Raises on failure.
    """
    import soundfile as sf

    audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    start_sample = int(onset  * sr)
    end_sample   = int(offset * sr)
    segment = audio[start_sample:end_sample]

    if sr != target_sr:
        n_out = int(len(segment) * target_sr / sr)
        segment = np.interp(
            np.linspace(0, len(segment) - 1, n_out),
            np.arange(len(segment)),
            segment,
        ).astype(np.float32)

    return segment


# ── frame index helpers ───────────────────────────────────────────────────────

def phoneme_frame_indices(onset: float, offset: float, n_frames: int) -> list[int]:
    """
    Return the list of frame indices (0-based) that overlap with
    the phoneme interval [onset, offset] given the model output
    has n_frames frames at XLSR_FRAME_RATE.
    """
    frame_start = max(0, math.floor(onset  * XLSR_FRAME_RATE))
    frame_end   = min(n_frames, math.ceil(offset * XLSR_FRAME_RATE))
    indices = list(range(frame_start, frame_end))
    # Fallback: use all frames if interval mapping produces empty set
    if not indices:
        indices = list(range(n_frames))
    return indices


# ── extraction ────────────────────────────────────────────────────────────────

def extract_xlsr_embeddings(
    manifest: list[dict],
    params: dict,
    device: str,
) -> tuple[np.ndarray, list[dict], list[int]]:
    """
    Returns:
        embeddings – (N, n_layers, D) float32
        meta       – list of N dicts
        layers     – list of layer indices (1-indexed as in the brief)
    """
    from transformers import Wav2Vec2Model, Wav2Vec2Processor
    import torch

    xp         = params["xlsr"]
    model_name = xp["model"]
    target_sr  = xp["target_sr"]
    layers     = xp["layers"]   # e.g. [6, 12, 18]  (1-indexed)
    pooling    = xp["pooling"]  # mean

    log.info(f"Loading model '{model_name}' …")
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model     = Wav2Vec2Model.from_pretrained(model_name, output_hidden_states=True)
    model.eval().to(device)
    log.info(f"Model loaded. Running on {device}.")

    embeddings = []
    meta       = []
    n_skipped  = 0
    N          = len(manifest)

    for i, row in enumerate(manifest):
        if i % 200 == 0:
            log.info(f"  [{i}/{N}] {row['speaker']} / {row['phoneme']}")

        try:
            audio = load_audio_segment(
                row["wav_path"], row["onset"], row["offset"], target_sr
            )
        except Exception as e:
            log.warning(f"  Audio load failed for token {i}: {e}")
            n_skipped += 1
            continue

        if len(audio) < XLSR_MIN_SAMPLES:
            log.debug(f"  Segment too short ({len(audio)} samples), skipping token {i}.")
            n_skipped += 1
            continue

        import torch
        inputs = processor(
            audio,
            sampling_rate=target_sr,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # hidden_states: tuple of (n_layers+1) tensors, each (1, T, D)
        # Index 0 = CNN feature extractor output, 1..n = transformer layers
        n_frames = outputs.hidden_states[0].shape[1]

        frame_idx = phoneme_frame_indices(row["onset"], row["offset"], n_frames)

        layer_vecs = []
        for layer_idx in layers:
            # layers param is 1-indexed transformer layers;
            # hidden_states[0] is CNN output, so transformer layer k → index k
            h = outputs.hidden_states[layer_idx]  # (1, T, D)
            h = h.squeeze(0).cpu().numpy()         # (T, D)
            h_phoneme = h[frame_idx]               # (|T_phoneme|, D)

            if pooling == "mean":
                vec = h_phoneme.mean(axis=0)
            elif pooling == "max":
                vec = h_phoneme.max(axis=0)
            else:
                vec = h_phoneme.mean(axis=0)

            layer_vecs.append(vec.astype(np.float32))

        embeddings.append(np.stack(layer_vecs, axis=0))  # (n_layers, D)
        meta.append({
            "speaker":     row["speaker"],
            "sentence_id": row["sentence_id"],
            "repetition":  row["repetition"],
            "phoneme":     row["phoneme"],
            "onset":       row["onset"],
            "offset":      row["offset"],
            "duration_ms": row["duration_ms"],
            "l1_status":   row["l1_status"],
            "gender":      row["gender"],
        })

    log.info(
        f"Extracted {len(embeddings)} embeddings, skipped {n_skipped} tokens."
    )
    return np.stack(embeddings, axis=0), meta, layers  # (N, n_layers, D)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true", help="Force CUDA device")
    args = parser.parse_args()

    import torch
    device = "cuda" if (args.gpu or torch.cuda.is_available()) else "cpu"

    params    = load_params()
    xp        = params["xlsr"]
    out_dir   = Path(xp["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(Path(params["data"]["processed_dir"]))
    log.info(f"Manifest: {len(manifest)} phoneme tokens.")

    t0 = time.time()
    embeddings, meta, layers = extract_xlsr_embeddings(manifest, params, device)
    elapsed = time.time() - t0

    # Save
    out_path = out_dir / "features_xlsr.npz"
    np.savez_compressed(
        out_path,
        embeddings=embeddings,             # (N, n_layers, D)
        layers=np.array(layers),           # (n_layers,)
        meta_json=np.array(json.dumps(meta, ensure_ascii=False)),
        info_json=np.array(json.dumps({
            "model":   xp["model"],
            "layers":  layers,
            "pooling": xp["pooling"],
            "shape":   list(embeddings.shape),
        }, ensure_ascii=False)),
    )
    log.info(f"Saved {out_path}  shape={embeddings.shape}  dtype={embeddings.dtype}")

    stats = {
        "n_tokens":        len(meta),
        "n_layers":        len(layers),
        "layers":          layers,
        "embedding_dim":   int(embeddings.shape[2]),
        "model":           xp["model"],
        "device":          device,
        "extraction_time_s": round(elapsed, 1),
    }
    with open(out_dir / "extraction_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    log.info(f"Done in {elapsed:.0f}s.")


if __name__ == "__main__":
    main()
