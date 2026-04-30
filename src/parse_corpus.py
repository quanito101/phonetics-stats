"""
Stage 1 – parse_corpus.py
──────────────────────────
Reads every TextGrid in data/raw/<SPEAKER>/ and extracts the
"phones" tier, producing one row per phoneme token:

    speaker, sentence_id, repetition, phoneme, onset, offset,
    duration_ms, l1_status, gender, wav_path

Speaker metadata (L1, gender) is read from metadata_RUFR.csv.

The filename convention is:
    <spk_lower>_<l1>_list<n>_FRcorp<m>.TextGrid
e.g. ab_rus_list1_FRcorp1.TextGrid  →  speaker AB, sentence FRcorp1

Because each sentence appears only once per recording file (not
multiple repetitions in one file), repetition index is derived
from the sentence number within a (speaker, sentence_id) group
after collecting all tokens.

Output
──────
data/processed/phonemes.csv          – one row per phoneme token
data/processed/corpus_stats.json     – quick summary for DVC metrics

Usage (called by DVC):
    python src/parse_corpus.py
"""

import csv
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── params ────────────────────────────────────────────────────────────────────

def load_params(path="params.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


# ── metadata ──────────────────────────────────────────────────────────────────

def load_speaker_metadata(raw_dir: Path) -> dict:
    """
    Read metadata_RUFR.csv → dict keyed by uppercase speaker code.
    Returns e.g. {"AB": {"l1_status": "L2", "gender": "F"}, ...}

    The CSV uses ';' as delimiter. Column names vary slightly across
    corpus versions so we do case-insensitive key matching.
    """
    path = raw_dir / "metadata_RUFR.csv"
    metadata = {}
    with open(path, encoding="utf-8-sig") as f:
        # Detect delimiter
        sample = f.read(1024)
        f.seek(0)
        delimiter = ";" if sample.count(";") > sample.count(",") else ","
        reader = csv.DictReader(f, delimiter=delimiter)
        # Normalise header keys to lowercase
        for raw_row in reader:
            row = {k.strip().lower(): v.strip() for k, v in raw_row.items()}

            # Speaker code – try common column names
            spk = (row.get("spk") or row.get("speaker") or row.get("code") or "").upper()
            if not spk:
                continue

            # L1 status: native French → L1, native Russian → L2
            l1_raw = (row.get("l1") or row.get("langue") or row.get("language") or "").lower()
            if "fr" in l1_raw or l1_raw == "l1":
                l1_status = "L1"
            elif "ru" in l1_raw or l1_raw == "l2":
                l1_status = "L2"
            else:
                l1_status = l1_raw.upper()

            # Gender
            gender_raw = (row.get("gender") or row.get("genre") or row.get("sex") or "").upper()
            gender = "F" if gender_raw.startswith("F") else ("M" if gender_raw.startswith("M") else gender_raw)

            metadata[spk] = {"l1_status": l1_status, "gender": gender}

    log.info(f"Loaded metadata for {len(metadata)} speakers.")
    return metadata


# ── TextGrid parser ───────────────────────────────────────────────────────────

def parse_textgrid(path: Path) -> dict[str, list[tuple]]:
    """
    Minimal hand-written TextGrid parser (no external dependency).
    Returns a dict  tier_name → [(xmin, xmax, text), ...]
    covering all non-empty intervals in each IntervalTier.

    Handles both the standard Praat short format and the long format
    shown in the corpus (with explicit field labels).
    """
    tiers = {}
    current_tier = None
    current_intervals = []
    xmin = xmax = text = None

    with open(path, encoding="utf-8-sig", errors="replace") as f:
        lines = [l.rstrip("\n") for l in f]

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # New tier
        if line.startswith('name = "'):
            # Save previous tier
            if current_tier is not None:
                tiers[current_tier] = current_intervals
            current_tier = line.split('"')[1]
            current_intervals = []
            xmin = xmax = text = None

        elif line.startswith("xmin =") and current_tier is not None:
            try:
                xmin = float(line.split("=")[1].strip())
            except ValueError:
                pass

        elif line.startswith("xmax =") and current_tier is not None:
            try:
                xmax = float(line.split("=")[1].strip())
            except ValueError:
                pass

        elif line.startswith("text =") and current_tier is not None:
            # text may span multiple lines (rare but possible)
            text_raw = line[len("text ="):].strip()
            # Strip surrounding quotes
            if text_raw.startswith('"'):
                text_raw = text_raw[1:]
                while not text_raw.endswith('"') and i + 1 < len(lines):
                    i += 1
                    text_raw += "\n" + lines[i].strip()
                text_raw = text_raw.rstrip('"')
            text = text_raw.strip()

            # Commit interval when all three fields are present
            if xmin is not None and xmax is not None:
                if text:  # skip silence/empty intervals
                    current_intervals.append((xmin, xmax, text))
                xmin = xmax = text = None

        i += 1

    # Save last tier
    if current_tier is not None:
        tiers[current_tier] = current_intervals

    return tiers


# ── filename parser ───────────────────────────────────────────────────────────

def parse_filename(stem: str) -> tuple[str, str] | None:
    """
    Extract (speaker_code, sentence_id) from a TextGrid stem.
    Pattern: <spk_lower>_<anything>_FRcorp<N>
    e.g. "ab_rus_list1_FRcorp1" → ("AB", "FRcorp1")
    """
    m = re.search(r"_([Ff][Rr][Cc]orp\d+)$", stem)
    if not m:
        return None
    sentence_id = m.group(1)
    # Speaker is the first token (before first underscore), uppercased
    speaker = stem.split("_")[0].upper()
    return speaker, sentence_id


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    params = load_params()
    raw_dir = Path(params["data"]["raw_dir"])
    out_dir = Path(params["data"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    speaker_meta = load_speaker_metadata(raw_dir)

    rows = []
    skipped_files = 0
    skipped_phonemes = 0
    unknown_speakers = set()

    # Walk every TextGrid in every speaker subfolder
    textgrids = sorted(raw_dir.rglob("*.TextGrid"))
    log.info(f"Found {len(textgrids)} TextGrid files.")

    # Track per-(speaker, sentence_id) repetition index
    rep_counter: dict[tuple, int] = defaultdict(int)

    for tg_path in textgrids:
        parsed = parse_filename(tg_path.stem)
        if parsed is None:
            log.warning(f"Cannot parse filename: {tg_path.name} – skipping.")
            skipped_files += 1
            continue

        speaker, sentence_id = parsed

        if speaker not in speaker_meta:
            unknown_speakers.add(speaker)
            skipped_files += 1
            continue

        meta = speaker_meta[speaker]
        wav_path = tg_path.with_suffix(".wav")
        if not wav_path.exists():
            log.warning(f"No WAV for {tg_path.name} – skipping.")
            skipped_files += 1
            continue

        tiers = parse_textgrid(tg_path)

        phones_tier = tiers.get("phones") or tiers.get("phone") or tiers.get("Phones")
        if phones_tier is None:
            log.warning(f"No 'phones' tier in {tg_path.name} – skipping.")
            skipped_files += 1
            continue

        key = (speaker, sentence_id)
        rep_counter[key] += 1
        rep_idx = rep_counter[key]

        for xmin, xmax, phoneme in phones_tier:
            duration_ms = (xmax - xmin) * 1000.0
            if duration_ms < 5.0:
                # Suspiciously short – likely an annotation artefact
                skipped_phonemes += 1
                continue

            rows.append({
                "speaker":     speaker,
                "sentence_id": sentence_id,
                "repetition":  rep_idx,
                "phoneme":     phoneme,
                "onset":       round(xmin, 6),
                "offset":      round(xmax, 6),
                "duration_ms": round(duration_ms, 3),
                "l1_status":   meta["l1_status"],
                "gender":      meta["gender"],
                "wav_path":    str(wav_path.resolve()),
            })

    if unknown_speakers:
        log.warning(f"Unknown speakers (no metadata): {sorted(unknown_speakers)}")

    log.info(
        f"Extracted {len(rows)} phoneme tokens from {len(textgrids) - skipped_files} files "
        f"({skipped_files} files skipped, {skipped_phonemes} short intervals dropped)."
    )

    # ── write CSV ─────────────────────────────────────────────────────────────
    out_csv = out_dir / "phonemes.csv"
    fieldnames = [
        "speaker", "sentence_id", "repetition", "phoneme",
        "onset", "offset", "duration_ms",
        "l1_status", "gender", "wav_path",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"Wrote {len(rows)} rows to {out_csv}")

    # ── quick stats ───────────────────────────────────────────────────────────
    speakers = {r["speaker"] for r in rows}
    phonemes = {r["phoneme"] for r in rows}
    l1_counts = defaultdict(int)
    gender_counts = defaultdict(int)
    for r in rows:
        l1_counts[r["l1_status"]] += 1
        gender_counts[r["gender"]] += 1

    stats = {
        "n_tokens":        len(rows),
        "n_speakers":      len(speakers),
        "n_phoneme_types": len(phonemes),
        "n_files_parsed":  len(textgrids) - skipped_files,
        "n_files_skipped": skipped_files,
        "n_short_dropped": skipped_phonemes,
        "l1_counts":       dict(l1_counts),
        "gender_counts":   dict(gender_counts),
        "phoneme_types":   sorted(phonemes),
    }
    with open(out_dir / "corpus_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    log.info(f"Stats: {stats['n_tokens']} tokens, {stats['n_speakers']} speakers, "
             f"{stats['n_phoneme_types']} phoneme types."

)
if __name__ == "__main__":
    main()
