# Fine-tuning Uni-Sign for 30-class Vietnamese isolated sign recognition

## 1. Objective and recommended direction

The target task is **Vietnamese isolated sign language recognition (ISLR)**: one trimmed video contains one Vietnamese word or short lexical label, and the model must choose one of 30 known labels.

The recommended first system is:

- Uni-Sign's existing pose-only ISLR architecture;
- warm-started from `pretrained_weight/unisign/wlasl_pose_only_islr.pth`;
- RTMPose/RTMLib whole-body poses with the 133-keypoint layout expected by this repository;
- signer-independent train/dev/test splits;
- mT5 generation during training, but **closed-vocabulary label scoring or constrained decoding** at evaluation and inference;
- progressive unfreezing and lower learning rates than the repository's generic stage-3 example.

Do not start with the RGB branch. The downloaded Uni-Sign checkpoint is pose-only and contains no RGB keys, while `fine_tuning.py` loads checkpoints with `strict=True`. Pose-only training is therefore the compatible, lower-risk baseline. Add RGB only as a later ablation after pose-only is reliable.

The current `data/CoSign/Ban ngày` folder can validate data preparation and overfitting, but a one-label run cannot measure a useful 30-class recognition result. Real model selection starts only after all 30 labels are present and every split contains every label.

## 2. What the current repository actually does

The relevant execution path is:

```text
fine_tuning.py
  -> S2T_Dataset in datasets.py
  -> load_part_kp: 133 whole-body keypoints -> body/left/right/face streams
  -> Uni_Sign in models.py
  -> spatial ST-GCN + temporal ST-GCN per body part
  -> pose_proj into 768-dimensional mT5 embeddings
  -> mT5 encoder/decoder generates the target text
  -> SLRT_metrics.islr_performance computes exact-match accuracy
```

Important consequences:

1. ISLR is currently implemented as text generation, not as a fixed 30-logit classifier. Adding labels does not require resizing a classifier head.
2. The mT5 tokenizer bundled in `pretrained_weight/mt5-base` supports Vietnamese text and preserves `Ban ngày` correctly.
3. `models.py` currently assigns `Chinese` only to datasets whose names contain `CSL`; every other dataset becomes `English`. CoSign must explicitly use `Vietnamese`.
4. The loader only supports `CSL_Daily`, `WLASL`, `How2Sign`, and `OpenASL`. `CoSign` must be added to the configuration, parser choices, and dataset path logic.
5. The loader expects gzip-pickled label dictionaries, despite filenames such as `labels.train` looking like plain text.
6. Pose files must contain `keypoints` and `scores`. Each frame must have shapes compatible with `(1, 133, 2)` and `(1, 133)`.
7. `S2T_Dataset.load_pose` derives a pose filename only by replacing `.mp4` with `.pkl`. It does not correctly handle `.avi`.
8. Frame selection uses `random.sample` for every phase. As written, dev and test predictions are nondeterministic when a clip is longer than `max_length`.
9. The repository command uses one optimizer learning rate for the entire pose encoder and mT5. The example `3e-4` is too aggressive for full fine-tuning a small 30-label dataset.
10. Training uses `drop_last=True`, which unnecessarily discards examples on a small dataset.
11. Free-form generation permits labels outside the 30-label vocabulary, and evaluation uses case- and Unicode-sensitive exact string equality.

These are adaptation requirements, not optional cleanup.

## 3. Audit of the supplied sample class

The inspected `Ban ngày` directory currently contains:

| Property | Observed value |
|---|---:|
| Total video files | 144 |
| MP4 files | 79 |
| AVI files | 65 |
| Directory size | about 737 MB |
| Approximate signer identities visible in filenames | 30 |
| Sample resolutions | 1280x720, 1920x1080, and 3840x2160 |
| Sample frame rates | 30 and 60 FPS |
| Sample lengths | roughly 111 to 368 frames |

The 65 AVI files are especially important: they represent 13 source recording IDs, each stored as five derived forms such as `cut`, `frames_35`, `frames_40`, `cut_frames_35`, and `cut_frames_40`. These are not 65 independent examples.

If derived variants of the same source appear in different splits, test accuracy will be badly inflated. Either keep one canonical variant or assign all variants through one `recording_id` group. The preferred policy is to retain one canonical, well-trimmed version and record the others in metadata as excluded derivatives.

The MP4 naming patterns appear to cover another roughly 17 signers, usually with multiple genuine repetitions. Confirm that interpretation against the data collection records rather than trusting filenames alone.

### Storage note

At the observed 737 MB per label, 30 equally sized raw label folders would use about 22 GB. The workspace filesystem currently has much less free space than its total capacity, so budget for raw videos, pose PKLs, experiment checkpoints, and caches before launching extraction. Avoid retaining several redundant encoded copies of every clip.

## 4. Canonical dataset design

Keep raw data immutable. Build a canonical manifest rather than making filenames the source of truth.

Suggested layout:

```text
data/CoSign/
├── raw/
│   ├── Ban ngày/
│   └── <29 other labels>/
├── pose_format/
│   ├── <sample_id>.pkl
│   └── ...
├── metadata/
│   ├── samples.csv
│   ├── labels.json
│   ├── signers.json
│   └── exclusions.csv
└── splits/
    ├── labels.train
    ├── labels.dev
    └── labels.test
```

It is also acceptable to leave raw label folders directly under `data/CoSign`; the manifest should still reference explicit paths and stable IDs.

### 4.1 Required manifest fields

Use at least these fields in `samples.csv`:

| Field | Meaning |
|---|---|
| `sample_id` | Stable ASCII identifier, never derived again after creation |
| `video_path` | UTF-8 relative path from the CoSign root |
| `pose_path` | Explicit relative PKL path; do not infer it by changing an extension |
| `label_id` | Stable integer from 0 to 29 |
| `label_text` | Canonical Vietnamese display label, NFC-normalized |
| `label_slug` | ASCII/internal label key |
| `signer_id` | Anonymized stable signer identity shared across all labels |
| `recording_id` | Identity of the original camera take |
| `variant_of` | Source recording for derived clips, empty for originals |
| `is_canonical` | Whether the sample is eligible for normal training/evaluation |
| `split` | `train`, `dev`, or `test`, assigned by signer |
| `fps`, `num_frames`, `width`, `height` | Media audit information |
| `pose_status` | `ok`, `low_confidence`, `missing`, or another controlled value |

Do not encode signer IDs or labels by repeatedly parsing inconsistent Unicode filenames at training time. Parse once, manually audit the result, and freeze the manifest.

### 4.2 Label normalization

Create one authoritative 30-label vocabulary. Apply exactly the same normalization to annotations, model outputs, and inference requests:

1. decode as UTF-8;
2. apply Unicode NFC normalization;
3. trim leading/trailing whitespace;
4. collapse repeated internal whitespace;
5. choose and consistently apply a casing policy;
6. preserve Vietnamese diacritics;
7. map aliases or alternate spellings only through an explicit alias table.

For example, choose exactly one of `Ban ngày` or `ban ngày` as the canonical target. Do not remove accents as a general evaluation rule: different Vietnamese words can collapse to the same ASCII form.

Validate that all 30 labels tokenize to non-empty mT5 sequences and store the token IDs in the vocabulary audit. This also determines the correct inference `max_new_tokens`; for isolated labels it should normally be the maximum label token length plus EOS, not 100.

### 4.3 Duplicate policy

Use the following hierarchy:

- Exact file checksum identifies byte-for-byte duplicates.
- `recording_id` identifies encodes/crops/resamples of the same take.
- `signer_id + label_id + repetition_id` identifies the same elicited repetition if multiple encodes exist.
- Manual review resolves ambiguous filename families.

Only canonical samples contribute to the primary metric. Derived variants may be used as an explicitly labeled training augmentation, but never as independent dev/test observations and never across signer/split boundaries.

## 5. Signer-independent splitting

The primary claim should be generalization to unseen signers. Therefore split by `signer_id`, globally across all 30 labels.

Recommended fixed split for about 30 signers:

- 20 signers for training;
- 5 signers for development;
- 5 signers for the final test set.

A 21/4/5 allocation is also reasonable if more training data is needed. The non-negotiable properties are:

- no signer occurs in more than one split;
- no `recording_id` occurs in more than one split;
- every one of the 30 labels occurs in all three splits;
- split assignments remain identical across experiments;
- the test split is not used for per-epoch model selection.

Use development accuracy for checkpoint selection and run the final test only after the recipe is frozen. For stronger research reporting, repeat training with three random seeds and/or run signer-group cross-validation on the train+dev signers while preserving the final five-signer test set.

If deployment will recognize known enrolled signers instead, report a separate signer-dependent result. Do not mix it with the primary unseen-signer result.

## 6. Pose extraction and quality control

### 6.1 Required extractor changes

`demo/pose_extraction.py` already produces the correct conceptual data:

```python
{
    "keypoints": [frame_keypoints, ...],
    "scores": [frame_scores, ...],
}
```

The CoSign implementation now changes the extraction workflow so that it:

- accepts both `mp4` and `avi`;
- constructs output names with `Path(video_path).stem + ".pkl"` or, preferably, the manifest `sample_id`;
- walks label directories or consumes the manifest instead of only using a non-recursive glob;
- cannot overwrite two files with the same basename from different label folders;
- records failures and pose statistics in a machine-readable report;
- uses one fixed RTMLib mode/model version for the entire dataset.

The current line `basename(...).replace(".mp4", ".pkl")` leaves an AVI output ending in `.avi`, which is incompatible with the training loader.

Prepare the manifest before training. The two CSV files are deliberate manual
inputs: heterogeneous filenames cannot safely establish signer identity or a
leak-free split automatically.

```bash
python script/prepare_cosign.py \
  --data-root data/CoSign \
  --signer-map data/CoSign/metadata/signer_map.csv \
  --split-map data/CoSign/metadata/split_map.csv \
  --expected-label-count 30
```

`signer_map.csv` must have `relative_path,signer_id` (and may include
`recording_id`); `split_map.csv` must have `signer_id,split`. The utility
writes `metadata/labels.json`, a human-readable sample manifest, and the
gzip-pickled `splits/labels.{train,dev,test}` files that Uni-Sign consumes.

Extract poses from the data root so the output preserves every label directory
and exactly matches manifest `pose_path` values:

```bash
PYTHONPATH=demo/rtmlib-main python demo/pose_extraction.py \
  --src_dir data/CoSign \
  --tgt_dir data/CoSign/pose_format \
  --device cuda \
  --backend onnxruntime \
  --mode performance \
  --video_extensions mp4 avi \
  --recursive
```

For the complete dataset, prefer a manifest-driven wrapper so output paths are stable. `performance` is a reasonable quality-first starting point on the available GPU, but test `performance` versus `lightweight` on a fixed subset before committing. The entire dataset must use the chosen mode; mixing extractors creates avoidable domain shift.

### 6.2 Pose QC gates

Reject or manually inspect samples that violate any of the following:

- unreadable video or zero decoded frames;
- mismatched number of `keypoints` and `scores` frames;
- keypoint shape other than `(1, 133, 2)` per frame;
- score shape other than `(1, 133)` per frame;
- NaN/Inf coordinates or scores;
- normalized coordinates substantially outside `[0, 1]` before Uni-Sign normalization;
- signer absent for a large fraction of frames;
- both hands below the repository's confidence threshold for most of the active sign;
- implausible identity jumps caused by detection switching;
- severe clipping of hands or face.

Produce per-sample summaries for mean body/left-hand/right-hand confidence, missing-hand fraction, decoded frames, and active interval. Visually inspect the worst samples and a random sample from every label and signer.

### 6.3 Temporal normalization

The inspected clips mix 30/60 FPS and contain roughly 111-368 frames. Normalize the temporal signal deliberately:

1. trim leading/trailing idle regions while retaining a small context margin;
2. define the active interval in metadata;
3. sample a fixed maximum sequence length from that interval;
4. preserve temporal order.

Start with `max_length=64`, matching the repository's WLASL ISLR example. Compare 64 and 96 frames if fast signs lose hand detail. For training, use stochastic but order-preserving temporal resampling. For dev/test, use deterministic uniform sampling. Never use `random.sample` during evaluation.

## 7. Required code adaptations

Implement these changes as a dedicated CoSign path rather than disguising the data as WLASL.

### 7.1 `config.py`

Add `CoSign` entries for all three split files, video root, and pose root. Prefer one root plus explicit paths from annotations:

```python
train_label_paths["CoSign"] = "./data/CoSign/splits/labels.train"
dev_label_paths["CoSign"] = "./data/CoSign/splits/labels.dev"
test_label_paths["CoSign"] = "./data/CoSign/splits/labels.test"
rgb_dirs["CoSign"] = "./data/CoSign"
pose_dirs["CoSign"] = "./data/CoSign/pose_format"
```

### 7.2 `utils.py`

- Add `CoSign` to `--dataset` choices.
- Add a `--language` option defaulting to `Vietnamese` for CoSign.
- Add fine-tuning controls such as `--freeze-mt5`, `--unfreeze-mt5-last-n`, differential learning rates, early-stopping patience, and deterministic evaluation sampling.
- Consider replacing the opaque gzip-pickle-only annotation API with JSONL/CSV ingestion. If compatibility is preferred, generate gzip-pickled dictionaries but retain human-readable manifests as the authoritative source.

### 7.3 `datasets.py`

Add a CoSign branch or, preferably, a `CoSignDataset` with these properties:

- consumes explicit `video_path` and `pose_path` fields;
- carries `label_id`, `signer_id`, and `recording_id` for auditing and grouped metrics;
- handles Unicode paths without string-based suffix assumptions;
- uses stochastic temporal sampling only for training;
- uses deterministic uniform temporal sampling for dev/test;
- refuses empty pose sequences before `collate_fn` accesses `vid[key][-1]`;
- validates the pose shapes once during preprocessing;
- uses `drop_last=False` in the training DataLoader;
- optionally balances batches across labels/signers without oversampling derivative duplicates.

If retaining the repository label-dictionary schema, each entry should minimally look like:

```python
{
    "cosign_000001": {
        "name": "cosign_000001",
        "video_path": "raw/Ban ngày/example.mp4",
        "pose_path": "pose_format/cosign_000001.pkl",
        "text": "Ban ngày",
        "gloss": ["Ban ngày"],
        "label_id": 0,
        "signer_id": "signer_001",
        "recording_id": "recording_000001"
    }
}
```

For `task=ISLR`, `text` is the canonical Vietnamese label. `gloss` is not used by the current ISLR path, but keeping it consistent avoids confusing diagnostics.

### 7.4 `models.py`

- Set the prompt language explicitly to Vietnamese. A suitable prompt is `Translate sign language video to Vietnamese:` to remain close to pretrained behavior.
- Do not infer language from whether the dataset name contains `CSL`.
- Add parameter-freezing helpers and optimizer parameter groups.
- Keep `hidden_dim=256` and the existing 768-dimensional `pose_proj`; they match the downloaded checkpoint.
- Add a closed-set inference method that scores each of the 30 canonical token sequences or constrains generation to those sequences.

For only 30 classes, scoring each label by conditional sequence log-likelihood is simple and robust:

```text
score(label | video) = sum of length-normalized decoder log-probabilities
prediction = label with highest score among the 30 labels
```

Batch the 30 candidate labels per video to keep inference efficient. This guarantees a valid label, supports top-k accuracy, and avoids free-generation spelling/case errors while preserving the pretrained generative model.

A 30-way classification head over pooled pose features is a useful secondary ablation, but it changes the checkpoint interface and loses some benefit of the multilingual decoder. Establish the generative/closed-vocabulary baseline first.

### 7.5 `fine_tuning.py`

- Load the pose-only checkpoint strictly for the pose-only model.
- Add a clean warm-start report that verifies missing/unexpected keys are zero.
- Use separate learning rates for newly adapted pose/alignment modules and mT5.
- save optimizer, scheduler, epoch, vocabulary, split checksum, and arguments if true training resume is required; the current checkpoints save model weights only;
- evaluate dev every epoch, but evaluate test only for the final selected checkpoint;
- implement early stopping;
- remove the first-sample-only 150-token padding workaround in `evaluate`;
- lower `max_new_tokens` to the vocabulary-derived maximum;
- normalize text before metrics;
- write prediction, reference, signer, confidence/score, and candidate ranking to a structured CSV/JSONL file.

For a one-GPU initial run, use one-GPU DeepSpeed. Multi-GPU evaluation in the current script is not a priority until data loading and metric gathering are made rank-safe.

### 7.6 `SLRT_metrics.py`

Keep the existing per-instance (`top1_acc_pi`) and per-class (`top1_acc_pc`) accuracies, but normalize both references and predictions first. Add:

- top-1 and top-5 accuracy from closed-label scores;
- macro precision, recall, and F1;
- balanced accuracy;
- 30x30 confusion matrix;
- per-label accuracy and support;
- per-signer accuracy;
- invalid-generation rate for the unconstrained-generation ablation.

Use macro/per-class accuracy as a primary checkpoint metric if class counts differ. Report per-instance accuracy as well for comparability with the repository.

## 8. Checkpoint loading plan

The available files have been verified:

- `pretrained_weight/mt5-base/pytorch_model.bin` and tokenizer files are present;
- `pretrained_weight/unisign/wlasl_pose_only_islr.pth` has a top-level `model` state dictionary;
- the Uni-Sign state dictionary has 627 keys and no RGB branch keys;
- its pose dimensions match the current default architecture;
- an actual CPU instantiation of the 587,747,368-parameter pose-only model loaded this checkpoint with `strict=True`, zero missing keys, and zero unexpected keys.

That same check confirmed that an unmodified model instantiated with `dataset='CoSign'` currently selects `English`, which is why the Vietnamese language adaptation above is required.

Use:

```text
pretrained_weight/unisign/wlasl_pose_only_islr.pth
```

as a **warm start**, not as a resumable optimizer checkpoint. Before the first long run, instantiate the CoSign pose-only model and assert that strict loading yields no missing or unexpected keys.

Do not pass `--rgb_support` with this checkpoint under the current strict loader. To add RGB later, load pose/mT5 keys strictly by subsystem, initialize RGB-specific keys separately, and print an explicit allowlist of missing RGB keys. Never hide arbitrary mismatch behind an unchecked `strict=False`.

## 9. Recommended training curriculum

### Phase 0: pipeline smoke test with `Ban ngày`

Purpose: validate extraction, Unicode, loading, forward/backward, checkpoint saving, and deterministic inference.

- Use only canonical videos.
- Create signer-grouped temporary train/dev splits.
- Overfit 8-16 training clips until training accuracy is near 100%.
- Confirm the decoded string exactly matches the canonical `Ban ngày` after normalization.
- Run evaluation twice and require identical predictions.

Do not report this as a recognition result; a one-class predictor is trivially accurate.

### Phase 1: frozen-mT5 alignment baseline

After all 30 labels pass readiness checks:

- initialize from WLASL pose-only ISLR;
- freeze all mT5 parameters initially;
- train pose/ST-GCN projection and alignment modules for about 3-5 epochs;
- use pose-only inputs, BF16, and an effective batch near 32;
- use pose/alignment LR around `1e-4`, weight decay around `1e-4` to `1e-2`, gradient clipping 1.0, and 5-10% warmup;
- start label smoothing at `0.0` or `0.05`, not the repository default `0.2`.

This phase adapts pose-domain statistics without immediately perturbing the full language model.

### Phase 2: progressive mT5 unfreezing

- unfreeze the last 1-2 mT5 encoder blocks, decoder layer norms/output path, or use LoRA if adding PEFT is acceptable;
- keep the pose/alignment LR around `5e-5` to `1e-4`;
- use an mT5 LR around `5e-6` to `2e-5`;
- continue with cosine decay and early stopping;
- select checkpoints by dev macro/per-class accuracy, with per-instance accuracy as a secondary metric.

If the frozen model cannot learn Vietnamese labels, unfreeze the decoder earlier. If training accuracy rises while signer-independent dev accuracy falls, reduce trainable mT5 capacity, strengthen pose augmentation, and stop earlier.

### Phase 3: conservative full-model fine-tuning

Only if Phase 2 underfits:

- unfreeze the full model;
- use a very low mT5 LR (`5e-6` to `1e-5`);
- retain a higher pose-branch LR through parameter groups;
- train for at most roughly 30-50 epochs with patience around 6-10 dev evaluations;
- keep and compare the best checkpoint, not merely the last checkpoint.

With a small dataset, longer full-model training is more likely to memorize signer/background cues than to improve generalization.

### Initial resource settings

For the available single 48 GB GPU, a sensible starting point is:

| Setting | Initial value |
|---|---:|
| Input | pose only |
| Precision | BF16 |
| `max_length` | 64 |
| Micro-batch | 8 (increase to 16 if stable) |
| Gradient accumulation | 4 (or 2 at micro-batch 16) |
| Effective batch | 32 |
| ZeRO stage | 2 |
| Gradient clipping | 1.0 |
| Warmup | 5-10% of optimizer steps |
| Label smoothing | 0.0 or 0.05 |
| Seeds | 42, 123, 2026 for final reporting |

Gradient accumulation does not increase BatchNorm's per-forward batch size, so prefer the largest stable micro-batch before increasing accumulation.

## 10. Augmentation policy

Start with pose augmentations that preserve label semantics:

- temporal shift/crop within the annotated active interval;
- mild speed perturbation through temporal resampling;
- small coordinate jitter relative to body scale;
- short contiguous frame masking;
- low-probability joint dropout guided by pose confidence;
- mild global translation/scale after the repository's coordinate normalization.

Avoid horizontal flipping initially. A safe flip must mirror x coordinates, swap all left/right keypoint indices and branches, and be known not to change the linguistic meaning. Validate it with a Vietnamese sign-language expert before enabling it.

Do not count cached crops or frame-rate variants as independent data augmentation evidence. Generate stochastic augmentation in training and retain the original `recording_id` for auditing.

## 11. Closed-vocabulary inference

Use two evaluation modes:

1. **Primary:** score all 30 canonical labels and select the maximum-likelihood label.
2. **Diagnostic:** retain the repository's unconstrained beam generation to measure invalid outputs and spelling behavior.

For the primary mode:

- pre-tokenize the 30 NFC-normalized labels;
- include EOS in every candidate;
- use the same target formatting as training;
- normalize scores by label token length or compare both normalized and unnormalized scoring on dev;
- return top-k candidates and normalized probabilities/scores;
- calibrate confidence on dev if downstream rejection is needed.

If the application must reject unknown signs, reserve explicit unknown/background videos and tune a rejection threshold. A 30-label softmax always chooses a known word and cannot detect out-of-vocabulary signs by itself.

## 12. Experiment matrix

Run experiments in this order so each addition has a clear justification:

| ID | Initialization | Trainable modules | Input | Decoder | Purpose |
|---|---|---|---|---|---|
| E0 | random | pose + mT5 | pose | closed label scoring | transfer-learning control |
| E1 | WLASL ISLR | pose/alignment only | pose | closed label scoring | safest pretrained baseline |
| E2 | WLASL ISLR | pose + last mT5 blocks | pose | closed label scoring | recommended main model |
| E3 | WLASL ISLR | full model, low LR | pose | closed label scoring | test whether more capacity helps |
| E4 | best E1-E3 | same | pose | unconstrained beam | quantify generation failures |
| E5 | best pose model + initialized RGB branch | staged | pose+RGB | closed label scoring | optional RGB gain |

Keep the split, canonical sample set, pose extractor, temporal sampling, and seeds fixed across this matrix.

## 13. Training command template

After the CoSign code adaptations and differential-LR arguments are implemented, the intended one-GPU command should resemble:

```bash
deepspeed --include localhost:0 --master_port 29511 fine_tuning.py \
  --dataset CoSign \
  --task ISLR \
  --language Vietnamese \
  --label-vocab data/CoSign/metadata/labels.json \
  --closed-vocabulary \
  --freeze-mt5 \
  --finetune pretrained_weight/unisign/wlasl_pose_only_islr.pth \
  --output_dir out/cosign_pose_islr_seed42 \
  --batch-size 8 \
  --gradient-accumulation-steps 4 \
  --epochs 50 \
  --max_length 64 \
  --opt AdamW \
  --lr 1e-4 \
  --mt5-lr 1e-5 \
  --warmup-epochs 3 \
  --label_smoothing 0.05 \
  --dtype bf16 \
  --zero_stage 2 \
  --seed 42
```

The CoSign implementation provides `--mt5-lr`, `--language`, freezing controls,
and closed-vocabulary scoring. Add the generated vocabulary to the command:

```text
--label-vocab data/CoSign/metadata/labels.json --closed-vocabulary --freeze-mt5
```

The command still requires the full 30-label dataset, complete signer maps,
pose files, and the repository dependencies. Do not add `--rgb_support` for
the primary pose-only experiment.

The final evaluation command should load `best_checkpoint.pth`, use deterministic frames and closed-label scoring, and write a structured prediction file. It should not update model state or inspect test results during hyperparameter selection.

## 14. Readiness gates

### Before pose extraction

- [ ] All 30 canonical labels and IDs are frozen.
- [ ] All signers have stable anonymized IDs across label folders.
- [ ] Derived recordings have a shared `recording_id`/`variant_of`.
- [ ] Train/dev/test signer lists are frozen.
- [ ] Available disk space is sufficient for pose files and experiments.

### Before training

- [ ] Every canonical video has exactly one readable pose PKL.
- [ ] Pose QC passes for all retained samples or exclusions are documented.
- [ ] Every label appears in every split.
- [ ] No signer or recording crosses splits.
- [ ] Unicode labels are NFC-normalized and tokenize correctly.
- [ ] Strict checkpoint load has zero missing/unexpected keys in pose-only mode.
- [ ] One-batch forward/backward succeeds.
- [ ] A tiny subset can be overfit.
- [ ] Two dev evaluations give identical outputs.

### Before final reporting

- [ ] Hyperparameters were selected without using test results.
- [ ] The best checkpoint is selected by a declared dev metric.
- [ ] Results include per-instance and macro/per-class accuracy.
- [ ] Per-label confusion and per-signer performance were inspected.
- [ ] At least three seeds or a signer-group confidence interval is reported.
- [ ] Split manifest, label vocabulary, code commit, checkpoint checksum, and full arguments are archived.

## 15. Failure diagnosis

| Symptom | Likely cause | First action |
|---|---|---|
| Checkpoint strict-load error with RGB keys | Pose-only checkpoint used with `--rgb_support` | Disable RGB for baseline |
| Missing AVI poses | `.replace(".mp4", ".pkl")` naming assumption | Use explicit `pose_path` and `Path.stem` |
| Very high dev accuracy immediately | signer/derived-recording leakage | Rebuild grouped splits and deduplicate |
| Evaluation changes between runs | random temporal sampling in dev/test | deterministic uniform sampling |
| Outputs are English or verbose | hard-coded language and free generation | set Vietnamese prompt and constrain labels |
| Correct word marked wrong | Unicode/case/whitespace mismatch | normalize both sides with the frozen policy |
| Training loss falls, unseen-signer accuracy does not | memorization of signer/background | freeze more mT5, strengthen pose augmentation, stop earlier |
| Hand features are mostly zero | low hand confidence, framing, or poor extraction | inspect QC and use higher-quality pose mode |
| Rare labels dominate errors | imbalance or insufficient signer coverage | macro metric, balanced sampling, collect more examples |
| Model always chooses a valid but wrong label confidently | closed-set model sees unknown/ambiguous signs | add unknown/background data and rejection calibration |

## 16. Recommended definition of success

The first successful milestone is not a long training run. It is a reproducible, signer-independent pose-only baseline for all 30 labels with:

- zero split leakage;
- deterministic evaluation;
- strict checkpoint compatibility;
- closed-vocabulary predictions;
- macro and per-instance metrics;
- archived manifests and experiment configuration.

Only after that baseline is stable should effort move to RGB fusion, full-model fine-tuning, or deployment-oriented unknown-sign rejection.
