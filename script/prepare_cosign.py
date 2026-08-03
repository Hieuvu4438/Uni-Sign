#!/usr/bin/env python3
"""Build stable CoSign manifests and Uni-Sign split annotations.

The raw CoSign collection is organised as one directory per Vietnamese label.
This script never moves, renames, or modifies source videos.  It writes a
human-readable manifest and gzip-pickled split dictionaries consumed by the
existing Uni-Sign loader.

Signer identity cannot be inferred safely from heterogeneous filenames.  Pass
``--signer-map`` and ``--split-map`` before creating train/dev/test files.
"""

import argparse
import csv
import gzip
import json
import pickle
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
RESERVED_ROOT_DIRS = {'pose_format', 'metadata', 'splits', '__pycache__'}


def normalize_label(value):
    value = unicodedata.normalize('NFC', str(value))
    return re.sub(r'\s+', ' ', value.strip())


def slugify(value):
    decomposed = unicodedata.normalize('NFKD', normalize_label(value))
    ascii_value = decomposed.encode('ascii', 'ignore').decode('ascii').lower()
    slug = re.sub(r'[^a-z0-9]+', '-', ascii_value).strip('-')
    return slug or 'label'


def normalize_recording_stem(stem):
    """Group known render/crop variants but preserve genuine repetitions."""
    value = unicodedata.normalize('NFC', stem).strip()
    value = re.sub(r'_cut_frames_(?:35|40)$', '', value, flags=re.IGNORECASE)
    value = re.sub(r'_frames_(?:35|40)$', '', value, flags=re.IGNORECASE)
    value = re.sub(r'_cut$', '', value, flags=re.IGNORECASE)
    return value


def load_csv_mapping(path, key_column):
    if not path:
        return {}
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or key_column not in reader.fieldnames:
            raise ValueError(f'{path} must include a {key_column!r} column')
        result = {}
        for row in reader:
            key = row[key_column].replace('\\', '/').strip()
            if not key:
                continue
            if key in result:
                raise ValueError(f'Duplicate {key_column} {key!r} in {path}')
            result[key] = {name: value.strip() for name, value in row.items() if value is not None}
    return result


def is_preferred_variant(path):
    stem = path.stem.lower()
    # Prefer a compact, cut source only when the alternatives are known derived
    # encodes.  The decision is recorded in the manifest and remains editable.
    if '_cut_frames_35' in stem:
        return (0, path.name)
    if '_cut_frames_40' in stem:
        return (1, path.name)
    if '_cut' in stem:
        return (2, path.name)
    if '_frames_35' in stem:
        return (3, path.name)
    if '_frames_40' in stem:
        return (4, path.name)
    return (5, path.name)


def discover_videos(video_root):
    videos = []
    for label_dir in sorted(video_root.iterdir()):
        if not label_dir.is_dir() or label_dir.name in RESERVED_ROOT_DIRS:
            continue
        for path in sorted(label_dir.rglob('*')):
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
                videos.append(path)
    return videos


def build_records(data_root, video_root, signer_map):
    videos = discover_videos(video_root)
    if not videos:
        raise ValueError(f'No videos found below {video_root}')

    labels = sorted({normalize_label(path.relative_to(video_root).parts[0]) for path in videos})
    label_id = {label: index for index, label in enumerate(labels)}
    used_slugs = Counter()
    label_rows = []
    for label in labels:
        base_slug = slugify(label)
        used_slugs[base_slug] += 1
        slug = base_slug if used_slugs[base_slug] == 1 else f'{base_slug}-{used_slugs[base_slug]}'
        label_rows.append({'id': label_id[label], 'text': label, 'slug': slug})

    grouped = defaultdict(list)
    for path in videos:
        relative_video = path.relative_to(video_root).as_posix()
        relative_data = path.relative_to(data_root).as_posix()
        label = normalize_label(path.relative_to(video_root).parts[0])
        mapping = signer_map.get(relative_video, {})
        recording_id = mapping.get('recording_id') or (
            f'{label_id[label]:02d}_{normalize_recording_stem(path.stem)}')
        grouped[(label, recording_id)].append((path, mapping))

    records = []
    sample_index = 0
    for (label, recording_id), entries in sorted(grouped.items(), key=lambda item: (label_id[item[0][0]], item[0][1])):
        preferred_path = min((path for path, _ in entries), key=is_preferred_variant)
        for path, mapping in sorted(entries, key=lambda item: item[0].as_posix()):
            relative_video = path.relative_to(video_root).as_posix()
            relative_data = path.relative_to(data_root).as_posix()
            sample_id = f'cosign_{label_id[label]:02d}_{sample_index:05d}'
            sample_index += 1
            pose_relative = (Path('pose_format') / Path(relative_video)).with_suffix('.pkl').as_posix()
            records.append({
                'sample_id': sample_id,
                'name': sample_id,
                'video_path': relative_data,
                'pose_path': pose_relative,
                'label_id': label_id[label],
                'label_text': label,
                'text': label,
                'gloss': [label],
                'signer_id': mapping.get('signer_id', 'UNASSIGNED') or 'UNASSIGNED',
                'recording_id': recording_id,
                'variant_of': '' if path == preferred_path else recording_id,
                'is_canonical': path == preferred_path,
                'source_size_bytes': path.stat().st_size,
            })
    return records, label_rows


def apply_splits(records, split_map):
    for record in records:
        signer = record['signer_id']
        record['split'] = split_map.get(signer, {}).get('split', 'unassigned')


def validate(records, labels, expected_label_count, allow_incomplete):
    errors = []
    if expected_label_count and len(labels) != expected_label_count:
        errors.append(f'expected {expected_label_count} labels, found {len(labels)}')
    canonical = [record for record in records if record['is_canonical']]
    if not canonical:
        errors.append('no canonical records')

    signer_splits = defaultdict(set)
    recording_splits = defaultdict(set)
    label_splits = defaultdict(set)
    for record in canonical:
        split = record['split']
        if split in {'train', 'dev', 'test'}:
            signer_splits[record['signer_id']].add(split)
            recording_splits[record['recording_id']].add(split)
            label_splits[record['label_text']].add(split)
    leaked_signers = [key for key, values in signer_splits.items() if len(values) > 1]
    leaked_recordings = [key for key, values in recording_splits.items() if len(values) > 1]
    if leaked_signers:
        errors.append(f'signer leakage across splits: {leaked_signers[:5]}')
    if leaked_recordings:
        errors.append(f'recording leakage across splits: {leaked_recordings[:5]}')
    missing_coverage = [label for label in (item['text'] for item in labels)
                        if label_splits[label] != {'train', 'dev', 'test'}]
    if missing_coverage:
        errors.append('labels missing train/dev/test coverage: ' + ', '.join(missing_coverage[:5]))
    if errors and not allow_incomplete:
        raise ValueError('; '.join(errors))
    return errors


def write_outputs(output_root, records, labels):
    metadata_dir = output_root / 'metadata'
    splits_dir = output_root / 'splits'
    metadata_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    with (metadata_dir / 'labels.json').open('w', encoding='utf-8') as f:
        json.dump({'labels': labels}, f, ensure_ascii=False, indent=2)

    with (metadata_dir / 'samples.jsonl').open('w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    fieldnames = list(records[0].keys())
    with (metadata_dir / 'samples.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = dict(record)
            row['gloss'] = json.dumps(row['gloss'], ensure_ascii=False)
            writer.writerow(row)

    for split in ('train', 'dev', 'test'):
        split_records = {
            record['sample_id']: {
                'name': record['name'],
                'video_path': record['video_path'],
                'pose_path': record['pose_path'],
                'text': record['text'],
                'gloss': record['gloss'],
                'label_id': record['label_id'],
                'signer_id': record['signer_id'],
                'recording_id': record['recording_id'],
            }
            for record in records
            if record['is_canonical'] and record['split'] == split
        }
        with gzip.open(splits_dir / f'labels.{split}', 'wb') as f:
            pickle.dump(split_records, f)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default='data/CoSign', type=Path)
    parser.add_argument('--video-root', default='',
                        help='Directory containing label folders, relative to --data-root (for example raw).')
    parser.add_argument('--signer-map', default='',
                        help='CSV: relative_path,signer_id[,recording_id]')
    parser.add_argument('--split-map', default='',
                        help='CSV: signer_id,split where split is train/dev/test')
    parser.add_argument('--expected-label-count', default=30, type=int)
    parser.add_argument('--include-derived', action='store_true',
                        help='Include derived encodes in split files (not recommended).')
    parser.add_argument('--allow-incomplete', action='store_true',
                        help='Write manifests even before all labels/splits are ready.')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    video_root = (data_root / args.video_root).resolve() if args.video_root else data_root
    if data_root not in video_root.parents and video_root != data_root:
        raise ValueError('--video-root must be inside --data-root')
    signer_map = load_csv_mapping(args.signer_map, 'relative_path')
    split_map = load_csv_mapping(args.split_map, 'signer_id')
    records, labels = build_records(data_root, video_root, signer_map)
    apply_splits(records, split_map)
    if args.include_derived:
        for record in records:
            record['is_canonical'] = True
    errors = validate(records, labels, args.expected_label_count, args.allow_incomplete)

    summary = {
        'data_root': str(data_root),
        'video_root': str(video_root),
        'labels': len(labels),
        'videos': len(records),
        'canonical_videos': sum(record['is_canonical'] for record in records),
        'signers': len({record['signer_id'] for record in records}),
        'split_counts': dict(Counter(record['split'] for record in records if record['is_canonical'])),
        'warnings': errors,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not args.dry_run:
        write_outputs(data_root, records, labels)
        print(f'Wrote metadata and splits below {data_root}')


if __name__ == '__main__':
    main()
