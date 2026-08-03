#!/usr/bin/env python3
"""Auto-generate signer_map.csv and split_map.csv from CoSign video filenames.

The CoSign filenames follow two patterns:

  MP4 (upper-case, named signers):
    LABEL_FIRSTNAME_LASTNAME_ (N).mp4
    LABEL_FIRSTNAME_LASTNAME_N.mp4

  AVI (mixed, from mobile recording app):
    FullName_age_label_timestamp_0_cut.avi
    FullName_age_label_timestamp_0_frames_35.avi
    FullName_age_label_timestamp_0_frames_40.avi
    FullName_age_label_timestamp_0_cut_frames_35.avi

This script extracts a best-effort signer name from each filename,
assigns a stable anonymized signer_id, and proposes a signer-independent
train/dev/test split (80/10/10 by signer count).

Output:
  data/CoSign/metadata/signer_map.csv
  data/CoSign/metadata/split_map.csv

IMPORTANT: Review both CSVs before running prepare_cosign.py.
The automatic name extraction is best-effort. Correct any wrong
signer_ids manually — especially AVI files where the name before
the underscore is a short nickname rather than the full name.
"""

import argparse
import csv
import os
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
RESERVED = {'pose_format', 'metadata', 'splits', '__pycache__'}

# Variant suffixes to strip before extracting signer identity
VARIANT_SUFFIXES = re.compile(
    r'(_cut_frames_(?:35|40)|_frames_(?:35|40)|_cut)$', re.IGNORECASE)

# AVI pattern: FullName_age_label_timestamp_0_...
AVI_PATTERN = re.compile(r'^(.+?)_(\d{1,3})_(.+?)_(\d{10,})', re.UNICODE)

# MP4 pattern: LABEL_NAME_NAME_... (N).mp4  or  LABEL_NAME_NAME..._N.mp4
# We strip the leading LABEL_ prefix (first segment in ALL-CAPS or matching label)
def extract_signer_from_mp4(stem: str, label_slug: str) -> str:
    """Remove the label prefix from an upper-case MP4 stem, return the signer portion."""
    # Normalise label to upper-case slug for matching
    label_upper = re.sub(r'[^A-Z0-9]', '_', label_slug.upper())
    # Remove leading label prefix
    cleaned = re.sub(rf'^{re.escape(label_upper)}_?', '', stem, count=1, flags=re.IGNORECASE)
    # Remove trailing index like _ (3) or _4
    cleaned = re.sub(r'_?\s*\(\s*\d+\s*\)\s*$', '', cleaned)
    cleaned = re.sub(r'_?\d+\s*$', '', cleaned)
    cleaned = cleaned.strip('_ ')
    return cleaned if cleaned else stem


def slugify_ascii(value: str) -> str:
    nfkd = unicodedata.normalize('NFKD', value)
    ascii_val = nfkd.encode('ascii', 'ignore').decode('ascii').upper()
    return re.sub(r'[^A-Z0-9]+', '_', ascii_val).strip('_') or 'UNKNOWN'


def extract_signer_name(path: Path, label_dir: str) -> str:
    stem = VARIANT_SUFFIXES.sub('', path.stem)
    ext = path.suffix.lower()

    if ext == '.avi':
        m = AVI_PATTERN.match(stem)
        if m:
            return m.group(1).strip()
        # Fallback: take everything before the first underscore-digit sequence
        before = re.split(r'_\d', stem)[0]
        return before.strip() if before else stem

    # MP4: strip label prefix
    label_slug = slugify_ascii(label_dir)
    return extract_signer_from_mp4(stem, label_slug)


def discover_videos(video_root: Path):
    """Yield (relative_path, label_dir, video_path) for all videos."""
    for label_dir in sorted(video_root.iterdir()):
        if not label_dir.is_dir() or label_dir.name in RESERVED:
            continue
        for vp in sorted(label_dir.rglob('*')):
            if vp.is_file() and vp.suffix.lower() in VIDEO_EXTENSIONS:
                yield vp.relative_to(video_root).as_posix(), label_dir.name, vp


def build_signer_map(video_root: Path):
    # Map raw extracted name -> canonical signer_id
    name_to_id: dict[str, str] = {}
    signer_counter = 1
    rows = []

    for rel_path, label_dir, vp in discover_videos(video_root):
        raw_name = extract_signer_name(vp, label_dir)
        # Normalise name for grouping (NFC + strip)
        norm_name = unicodedata.normalize('NFC', raw_name).strip()
        if not norm_name:
            norm_name = 'UNKNOWN'

        if norm_name not in name_to_id:
            sid = f'signer_{signer_counter:03d}'
            name_to_id[norm_name] = sid
            signer_counter += 1

        rows.append({
            'relative_path': rel_path,
            'signer_id': name_to_id[norm_name],
            'extracted_name': norm_name,   # extra column for manual review
        })

    return rows, name_to_id


def build_split_map(name_to_id: dict[str, str],
                    train_frac: float = 0.80,
                    dev_frac: float = 0.10) -> list[dict]:
    """Assign signers to train/dev/test deterministically by sorted ID."""
    signer_ids = sorted(set(name_to_id.values()))
    n = len(signer_ids)
    n_train = max(1, round(n * train_frac))
    n_dev   = max(1, round(n * dev_frac))
    n_test  = max(1, n - n_train - n_dev)

    # Adjust so they sum exactly
    if n_train + n_dev + n_test > n:
        n_train = n - n_dev - n_test
    if n_train + n_dev + n_test < n:
        n_train += n - n_train - n_dev - n_test

    splits = (
        ['train'] * n_train +
        ['dev']   * n_dev   +
        ['test']  * n_test
    )

    return [
        {'signer_id': sid, 'split': sp}
        for sid, sp in zip(signer_ids, splits)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default='data/CoSign', type=Path,
                        help='Root of the CoSign dataset (default: data/CoSign)')
    parser.add_argument('--train-frac', default=0.80, type=float)
    parser.add_argument('--dev-frac',   default=0.10, type=float)
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing CSV files')
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    metadata_dir = data_root / 'metadata'
    metadata_dir.mkdir(parents=True, exist_ok=True)

    signer_map_path = metadata_dir / 'signer_map.csv'
    split_map_path  = metadata_dir / 'split_map.csv'

    if not args.overwrite:
        for p in (signer_map_path, split_map_path):
            if p.exists():
                print(f'[SKIP] {p} already exists. Use --overwrite to replace.')
                return

    print(f'Scanning videos under {data_root} ...')
    rows, name_to_id = build_signer_map(data_root)

    # Write signer_map.csv
    signer_fieldnames = ['relative_path', 'signer_id', 'extracted_name']
    with signer_map_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=signer_fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f'Wrote {len(rows)} entries -> {signer_map_path}')
    print(f'Unique signer names found: {len(name_to_id)}')

    # Write split_map.csv
    split_rows = build_split_map(name_to_id, args.train_frac, args.dev_frac)
    from collections import Counter
    split_counts = Counter(r['split'] for r in split_rows)
    print(f'Split distribution: {dict(split_counts)}')

    split_fieldnames = ['signer_id', 'split']
    with split_map_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=split_fieldnames)
        writer.writeheader()
        writer.writerows(split_rows)

    print(f'Wrote {len(split_rows)} signer entries -> {split_map_path}')
    print()
    print('NEXT STEPS:')
    print(f'  1. Review {signer_map_path} — check "extracted_name" column for grouping errors.')
    print(f'  2. Review {split_map_path} — ensure each split covers all 30 labels.')
    print(f'  3. Run:')
    print(f'       python script/prepare_cosign.py \\')
    print(f'         --data-root {args.data_root} \\')
    print(f'         --signer-map {signer_map_path} \\')
    print(f'         --split-map {split_map_path} \\')
    print(f'         --expected-label-count 30 \\')
    print(f'         --allow-incomplete')


if __name__ == '__main__':
    main()
