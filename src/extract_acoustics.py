"""
Stage 2 – extract_acoustics.py
───────────────────────────────
For every phoneme token in data/processed/phonemes.csv, extracts:

    F1, F2, F3   – formant frequencies at the midpoint (Hz)
                   for long vowels (> long_vowel_threshold_ms) also at 25% and 75%
    f0           – mean fundamental frequency over the interval (voiced only, Hz)
    duration_ms  – already in the manifest, copied through
    SCG          – spectral centre of gravity (fricatives only, Hz)

LPC parameters follow the project brief:
    max_formant = 5000 Hz for female speakers, 4500 Hz for male speakers
    n_formants  = 5

Missing values (formant tracker failures, unvoiced f0) are left as NaN
and reported in extraction_stats.json.

Output
──────
data/acoustics/features_acoustic.csv   – one row per phoneme token
data/acoustics/extraction_stats.json   – missing-value report

Usage (called by DVC):
    python src/extract_acoustics.py
"""

import csv
import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── phoneme class sets ────────────────────────────────────────────────────────
# French oral vowels targeted by the project brief
FRENCH_ORAL_VOWELS = {
    "a", "e", "i", "o", "u", "y",
    "ø", "œ", "ɛ", "ɔ", "ə", "ɑ",
    # common diacritic variants present in this corpus
    "ɑ̃",   # nasal – kept for completeness; treated as vowel for formants
    "ɛ̃", "œ̃", "ɔ̃",
}

# Nasal vowels (F3 less meaningful but F1/F2 still extracted)
NASAL_VOWELS = {"ɑ̃", "ɛ̃", "œ̃", "ɔ̃"}

# All vowels = oral + nasal
ALL_VOWELS = FRENCH_ORAL_VOWELS | NASAL_VOWELS

# Fricatives for SCG
FRICATIVES = {"f", "v", "s", "z", "ʃ", "ʒ", "χ", "ʁ"}


def is_vowel(phoneme: str) -> bool:
    # Strip diacritics for matching: keep base character
    base = phoneme[0] if phoneme else ""
    return phoneme in ALL_VOWELS or base in {p[0] for p in ALL_VOWELS}


def is_fricative(phoneme: str) -> bool:
    base = phoneme[0] if phoneme else ""
    return phoneme in FRICATIVES or base in FRICATIVES


# ── params ────────────────────────────────────────────────────────────────────

def load_params(path="params.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


# ── audio cache ───────────────────────────────────────────────────────────────
# Loading the same WAV file repeatedly is expensive; cache by path.
_sound_cache: dict[str, parselmouth.Sound] = {}
_CACHE_MAX = 5  # keep at most 5 files in memory at once


def get_sound(wav_path: str) -> parselmouth.Sound:
    if wav_path not in _sound_cache:
        if len(_sound_cache) >= _CACHE_MAX:
            # Evict oldest entry
            _sound_cache.pop(next(iter(_sound_cache)))
        _sound_cache[wav_path] = parselmouth.Sound(wav_path)
    return _sound_cache[wav_path]


# ── feature extraction ────────────────────────────────────────────────────────

def extract_formants_at_time(
    sound: parselmouth.Sound,
    t: float,
    max_formant: float,
    n_formants: int,
    window_length: float,
) -> tuple[float, float, float]:
    """Return (F1, F2, F3) at time t using Burg LPC. NaN on failure."""
    formant = call(
        sound, "To Formant (burg)",
        0.0,          # time step (0 = auto)
        n_formants,
        max_formant,
        window_length,
        50.0,         # pre-emphasis from (Hz)
    )
    def safe_get(n):
        v = call(formant, "Get value at time", n, t, "Hertz", "Linear")
        return float("nan") if (v is None or math.isnan(v) or v <= 0) else v

    return safe_get(1), safe_get(2), safe_get(3)


def extract_f0_mean(
    sound: parselmouth.Sound,
    onset: float,
    offset: float,
) -> float:
    """Mean f0 over [onset, offset] using autocorrelation. NaN if unvoiced or too short."""
    try:
        segment = sound.extract_part(from_time=onset, to_time=offset, preserve_times=True)
        pitch = segment.to_pitch_ac(
            time_step=0.01,
            pitch_floor=75.0,
            pitch_ceiling=600.0,
        )
        frames = [
            pitch.get_value_in_frame(i)
            for i in range(1, pitch.get_number_of_frames() + 1)
        ]
        voiced = [f for f in frames if f is not None and not math.isnan(f) and f > 0]
        return float(np.mean(voiced)) if voiced else float("nan")
    except Exception:
        return float("nan")


def extract_scg(
    sound: parselmouth.Sound,
    onset: float,
    offset: float,
) -> float:
    """Spectral centre of gravity over [onset, offset]. NaN on failure."""
    try:
        segment = sound.extract_part(from_time=onset, to_time=offset, preserve_times=False)
        spectrum = segment.to_spectrum()
        scg = call(spectrum, "Get centre of gravity", 2.0)
        return float("nan") if (scg is None or math.isnan(scg)) else float(scg)
    except Exception:
        return float("nan")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    params  = load_params()
    ap      = params["acoustics"]
    raw_dir = Path(params["data"]["raw_dir"])
    out_dir = Path(ap["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    n_formants      = ap["n_formants"]
    window_length   = ap["window_length"]
    long_threshold  = ap["long_vowel_threshold_ms"]
    max_f_female    = ap["formant_max_female"]
    max_f_male      = ap["formant_max_male"]

    # Load manifest
    manifest = pd.read_csv(
        Path(params["data"]["processed_dir"]) / "phonemes.csv",
        encoding="utf-8",
    )
    log.info(f"Manifest loaded: {len(manifest)} tokens.")

    rows = []
    missing = {"F1": 0, "F2": 0, "F3": 0, "f0": 0, "SCG": 0}
    n_long_vowels = 0

    for idx, row in manifest.iterrows():
        if idx % 500 == 0:
            log.info(f"  [{idx}/{len(manifest)}] {row['speaker']} / {row['phoneme']}")

        phoneme   = str(row["phoneme"])
        onset     = float(row["onset"])
        offset    = float(row["offset"])
        dur_ms    = float(row["duration_ms"])
        gender    = str(row["gender"])
        wav_path  = str(row["wav_path"])

        max_formant = max_f_female if gender == "F" else max_f_male
        midpoint    = (onset + offset) / 2.0

        # Initialise all features as NaN
        F1 = F2 = F3 = float("nan")
        F1_25 = F2_25 = F1_75 = F2_75 = float("nan")
        f0 = scg = float("nan")

        try:
            sound = get_sound(wav_path)
        except Exception as e:
            log.warning(f"Cannot load {wav_path}: {e}")
            rows.append(_make_row(row, F1, F2, F3, F1_25, F2_25, F1_75, F2_75, f0, scg))
            continue

        # ── formants (vowels only) ────────────────────────────────────────────
        if is_vowel(phoneme):
            F1, F2, F3 = extract_formants_at_time(
                sound, midpoint, max_formant, n_formants, window_length
            )
            if math.isnan(F1): missing["F1"] += 1
            if math.isnan(F2): missing["F2"] += 1
            if math.isnan(F3): missing["F3"] += 1

            # Trajectory points for long vowels
            if dur_ms > long_threshold:
                n_long_vowels += 1
                F1_25, F2_25, _ = extract_formants_at_time(
                    sound, onset + 0.25 * (offset - onset),
                    max_formant, n_formants, window_length,
                )
                F1_75, F2_75, _ = extract_formants_at_time(
                    sound, onset + 0.75 * (offset - onset),
                    max_formant, n_formants, window_length,
                )

        # ── f0 (all segments; NaN for unvoiced) ──────────────────────────────
        f0 = extract_f0_mean(sound, onset, offset)
        if math.isnan(f0):
            missing["f0"] += 1

        # ── SCG (fricatives only) ─────────────────────────────────────────────
        if is_fricative(phoneme):
            scg = extract_scg(sound, onset, offset)
            if math.isnan(scg):
                missing["SCG"] += 1

        rows.append(_make_row(row, F1, F2, F3, F1_25, F2_25, F1_75, F2_75, f0, scg))

    # ── write CSV ─────────────────────────────────────────────────────────────
    out_df = pd.DataFrame(rows)
    out_csv = out_dir / "features_acoustic.csv"
    out_df.to_csv(out_csv, index=False, encoding="utf-8")
    log.info(f"Wrote {len(out_df)} rows to {out_csv}")

    # ── missing value report ──────────────────────────────────────────────────
    n_vowels     = sum(1 for r in rows if not math.isnan(r.get("F1", float("nan")) or float("nan")))
    n_total      = len(rows)
    vowel_tokens = int(out_df["F1"].notna().sum() + out_df["F1"].isna().sum() *
                       (out_df["phoneme"].apply(is_vowel)).sum() / max(n_total, 1))

    # Per-phoneme missing rate
    vowel_df = out_df[out_df["phoneme"].apply(is_vowel)]
    per_phoneme_missing = (
        vowel_df.groupby("phoneme")[["F1", "F2", "F3"]]
        .apply(lambda g: g.isna().mean().to_dict())
        .to_dict()
    )

    stats = {
        "n_total_tokens":   n_total,
        "n_long_vowels":    n_long_vowels,
        "missing_counts":   missing,
        "missing_rates":    {k: round(v / max(n_total, 1), 4) for k, v in missing.items()},
        "per_phoneme_missing_F1": {
            p: round(d.get("F1", 0), 4)
            for p, d in per_phoneme_missing.items()
        },
    }
    with open(out_dir / "extraction_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    log.info(f"Missing values: {missing}")
    log.info(f"Long vowels (>{long_threshold}ms): {n_long_vowels}")


def _make_row(manifest_row, F1, F2, F3, F1_25, F2_25, F1_75, F2_75, f0, scg):
    return {
        # pass-through columns from manifest
        "speaker":     manifest_row["speaker"],
        "sentence_id": manifest_row["sentence_id"],
        "repetition":  manifest_row["repetition"],
        "phoneme":     manifest_row["phoneme"],
        "onset":       manifest_row["onset"],
        "offset":      manifest_row["offset"],
        "duration_ms": manifest_row["duration_ms"],
        "l1_status":   manifest_row["l1_status"],
        "gender":      manifest_row["gender"],
        "wav_path":    manifest_row["wav_path"],
        # acoustic features
        "F1":    F1,
        "F2":    F2,
        "F3":    F3,
        "F1_25": F1_25,
        "F2_25": F2_25,
        "F1_75": F1_75,
        "F2_75": F2_75,
        "f0":    f0,
        "SCG":   scg,
    }


if __name__ == "__main__":
    main()
