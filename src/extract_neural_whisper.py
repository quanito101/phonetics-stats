"""
Stage 4 – extract_neural_whisper.py
─────────────────────────────────────
Extracts Whisper encoder hidden states for every phoneme token
in data/processed/phonemes.csv.

Whisper processes log-mel spectrograms. The encoder operates at a
fixed frame rate of 50 Hz (hop = 160 samples at 16 kHz, after the
2× strided CNN). For each phoneme token, frames overlapping the
phoneme interval are mean-pooled into a single vector per layer.

Output
──────
data/neural/whisper/features_whisper.npz  containing:
    embeddings  – (N, n_layers, D) float32
    layers      – 1-D int array of layer indices extracted
    meta_json   – JSON string with token metadata
    info_json   – JSON string with model/extraction details

data/neural/whisper/extraction_stats.json

Note on Whisper input length
────────────────────────────
Whisper's encoder expects exactly 30 seconds of audio (3000 mel frames).
We pad/trim each individual WAV file to 30 s before encoding, then
index only the frames corresponding to the phoneme interval.
This is the standard approach when using Whisper as a feature extractor.

Usage:
    python src/extract_neural_whisper.py          # CPU (slow, for testing)
    python src/extract_neural_whisper.py --gpu    # Kaggle / GPU machine
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

# Whisper encoder frame rate after 2× CNN stride (16000 / 320 = 50 Hz)
WHISPER_FRAME_RATE = 50.0
WHISPER_N_MEL = 80
WHISPER_SAMPLE_RATE = 16000
WHISPER_CHUNK_SAMPLES = 30 * WHISPER_SAMPLE_RATE  # 480000 samples = 30 s


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


def load_full_audio(wav_path: str, target_sr: int) -> np.ndarray:
    """
    Load the entire WAV file (not just a segment) resampled to target_sr.
    Whisper needs the full-file context to produce a meaningful 30-s window.
    Returns 1-D float32 array.
    """
    import soundfile as sf

    audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if sr != target_sr:
        n_out = int(len(audio) * target_sr / sr)
        audio = np.interp(
            np.linspace(0, len(audio) - 1, n_out),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)

    return audio


def pad_or_trim(audio: np.ndarray, length: int) -> np.ndarray:
    """Pad with zeros or trim to exactly `length` samples."""
    if len(audio) >= length:
        return audio[:length]
    return np.pad(audio, (0, length - len(audio)))


def phoneme_frame_indices(onset: float, offset: float, n_frames: int) -> list[int]:
    frame_start = max(0, math.floor(onset  * WHISPER_FRAME_RATE))
    frame_end   = min(n_frames, math.ceil(offset * WHISPER_FRAME_RATE))
    indices = list(range(frame_start, frame_end))
    if not indices:
        indices = list(range(n_frames))
    return indices


# ── extraction ────────────────────────────────────────────────────────────────

def extract_whisper_embeddings(
    manifest: list[dict],
    params: dict,
    device: str,
) -> tuple[np.ndarray, list[dict], list[int]]:
    """
    Returns:
        embeddings – (N, n_layers, D) float32
        meta       – list of N dicts
        layers     – list of extracted layer indices (1-indexed)
    """
    import torch
    from transformers import WhisperModel, WhisperProcessor

    wp         = params["whisper"]
    model_name = wp["model"]
    target_sr  = wp["target_sr"]
    layers     = wp["layers"]   # e.g. [4, 20]  (1-indexed encoder layers)
    pooling    = wp["pooling"]

    log.info(f"Loading model '{model_name}' …")
    processor = WhisperProcessor.from_pretrained(model_name)
    model     = WhisperModel.from_pretrained(model_name, output_hidden_states=True)
    model.eval().to(device)
    log.info(f"Model loaded. Running on {device}.")

    # Cache: avoid reloading the same WAV for multiple phonemes
    _audio_cache: dict[str, np.ndarray] = {}

    # We also cache the encoder hidden states per WAV file to avoid
    # re-running the full encoder for every phoneme in the same recording.
    _encoder_cache: dict[str, tuple] = {}  # wav_path → hidden_states tuple

    embeddings = []
    meta       = []
    n_skipped  = 0
    N          = len(manifest)

    for i, row in enumerate(manifest):
        if i % 200 == 0:
            log.info(f"  [{i}/{N}] {row['speaker']} / {row['phoneme']}")

        wav_path = row["wav_path"]

        # ── load + encode full WAV (cached) ───────────────────────────────────
        if wav_path not in _encoder_cache:
            # Evict cache if it grows large (memory safety on Kaggle)
            if len(_encoder_cache) > 10:
                oldest = next(iter(_encoder_cache))
                del _encoder_cache[oldest]
                if oldest in _audio_cache:
                    del _audio_cache[oldest]

            try:
                audio = load_full_audio(wav_path, target_sr)
                _audio_cache[wav_path] = audio
            except Exception as e:
                log.warning(f"  Cannot load {wav_path}: {e}")
                n_skipped += 1
                continue

            audio_padded = pad_or_trim(audio, WHISPER_CHUNK_SAMPLES)

            # Build log-mel spectrogram input
            inputs = processor(
                audio_padded,
                sampling_rate=target_sr,
                return_tensors="pt",
            )
            input_features = inputs.input_features.to(device)  # (1, 80, 3000)

            # Whisper encoder forward pass
            # We need a dummy decoder input to get encoder hidden states
            with torch.no_grad():
                encoder_outputs = model.encoder(
                    input_features,
                    output_hidden_states=True,
                    return_dict=True,
                )
            # hidden_states: tuple of (n_layers+1) tensors (1, T_enc, D)
            # T_enc = 1500 for whisper-medium (3000 mel frames / 2 due to encoder conv)
            _encoder_cache[wav_path] = tuple(
                h.squeeze(0).cpu().numpy()  # (T_enc, D)
                for h in encoder_outputs.hidden_states
            )

        hidden_states = _encoder_cache[wav_path]
        n_frames      = hidden_states[0].shape[0]  # T_enc

        frame_idx = phoneme_frame_indices(row["onset"], row["offset"], n_frames)

        layer_vecs = []
        for layer_idx in layers:
            # hidden_states[0] = embedding output, [1..n] = transformer layers
            h = hidden_states[layer_idx]       # (T_enc, D)
            h_phoneme = h[frame_idx]            # (|T_phoneme|, D)

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
    return np.stack(embeddings, axis=0), meta, layers


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true", help="Force CUDA device")
    args = parser.parse_args()

    import torch
    device = "cuda" if (args.gpu or torch.cuda.is_available()) else "cpu"

    params  = load_params()
    wp      = params["whisper"]
    out_dir = Path(wp["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(Path(params["data"]["processed_dir"]))
    log.info(f"Manifest: {len(manifest)} phoneme tokens.")

    t0 = time.time()
    embeddings, meta, layers = extract_whisper_embeddings(manifest, params, device)
    elapsed = time.time() - t0

    out_path = out_dir / "features_whisper.npz"
    np.savez_compressed(
        out_path,
        embeddings=embeddings,
        layers=np.array(layers),
        meta_json=np.array(json.dumps(meta, ensure_ascii=False)),
        info_json=np.array(json.dumps({
            "model":   wp["model"],
            "layers":  layers,
            "pooling": wp["pooling"],
            "shape":   list(embeddings.shape),
        }, ensure_ascii=False)),
    )
    log.info(f"Saved {out_path}  shape={embeddings.shape}  dtype={embeddings.dtype}")

    stats = {
        "n_tokens":          len(meta),
        "n_layers":          len(layers),
        "layers":            layers,
        "embedding_dim":     int(embeddings.shape[2]),
        "model":             wp["model"],
        "device":            device,
        "extraction_time_s": round(elapsed, 1),
    }
    with open(out_dir / "extraction_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    log.info(f"Done in {elapsed:.0f}s.")


if __name__ == "__main__":
    main()
