# CoSign Dataset Specification: 30-Class Vietnamese Isolated Sign Language Recognition

## 1. Overview and Dataset Summary

**CoSign** is a benchmark dataset for **Vietnamese Isolated Sign Language Recognition (ISLR)**. It contains trimmed video recordings covering 30 high-frequency Vietnamese sign vocabulary classes collected across a diverse group of signers. Each video clip depicts a single Vietnamese sign performed by a signer.

The dataset is configured for training and evaluation with the **Uni-Sign** architecture using a pose-only multi-stream spatial-temporal graph convolutional network (ST-GCN) fused with a multilingual mT5 language backbone.

---

## 2. Dataset Paths and Storage Structure

### 2.1 Primary File System Locations

- **Dataset Root Directory:** `/home/cosign/Uni-Sign/data/CoSign`
- **Relative Path within Repository:** `data/CoSign`
- **Pose Cache Directory:** `data/CoSign/pose_format`
- **Metadata Directory:** `data/CoSign/metadata`
- **Split Annotations Directory:** `data/CoSign/splits`

### 2.2 Directory Layout

```text
data/CoSign/
├── Ban ngày/                  # Label directory: "Ban ngày" (Daytime)
│   ├── BAN_NGÀY_BÙI_MAI_CHI_ (1).mp4
│   └── ...
├── Ban đêm/                   # Label directory: "Ban đêm" (Night)
├── Bàn tay/                   # Label directory: "Bàn tay" (Hand)
├── Bạn thân/                  # Label directory: "Bạn thân" (Best friend)
├── Bệnh viện/                 # Label directory: "Bệnh viện" (Hospital)
├── Chiều/                     # Label directory: "Chiều" (Afternoon)
├── Chào/                      # Label directory: "Chào" (Hello)
├── Chân/                      # Label directory: "Chân" (Leg / Foot)
├── Chúng ta/                  # Label directory: "Chúng ta" (We / Us)
├── Chậm lại/                  # Label directory: "Chậm lại" (Slow down)
├── Con gấu/                   # Label directory: "Con gấu" (Bear)
├── Cá/                        # Label directory: "Cá" (Fish)
├── Có thể/                    # Label directory: "Có thể" (Can / Possible)
├── Cơ thể/                    # Label directory: "Cơ thể" (Body)
├── Cứu/                       # Label directory: "Cứu" (Save / Rescue)
├── Dễ/                        # Label directory: "Dễ" (Easy)
├── Hôm nay/                   # Label directory: "Hôm nay" (Today)
├── Họ/                        # Label directory: "Họ" (They / Them)
├── Học sinh/                  # Label directory: "Học sinh" (Student)
├── Khóc/                      # Label directory: "Khóc" (Cry)
├── Mua/                       # Label directory: "Mua" (Buy)
├── Mời vào/                   # Label directory: "Mời vào" (Welcome / Come in)
├── Nghe/                      # Label directory: "Nghe" (Listen / Hear)
├── Ngón tay/                  # Label directory: "Ngón tay" (Finger)
├── Nhà/                       # Label directory: "Nhà" (House / Home)
├── Nhìn/                      # Label directory: "Nhìn" (Look / See)
├── Nhầm/                      # Label directory: "Nhầm" (Mistake / Wrong)
├── Nói/                       # Label directory: "Nói" (Speak / Talk)
├── Nặng/                      # Label directory: "Nặng" (Heavy)
├── Ăn/                        # Label directory: "Ăn" (Eat)
├── pose_format/               # Extracted RTMPose Whole-body .pkl keypoint files
├── metadata/                  # Dataset manifests and vocabulary definitions
│   ├── labels.json            # Canonical 30-class Vietnamese label vocabulary
│   ├── samples.csv            # Sample manifest table
│   ├── samples.jsonl          # Sample manifest JSONL
│   ├── signer_map.csv         # Signer identity mapping
│   └── split_map.csv          # Signer-independent split assignment
└── splits/                    # Gzip-pickled split annotations for Uni-Sign loader
    ├── labels.train
    ├── labels.dev
    └── labels.test
```

---

## 3. Dataset Vocabulary and Class Statistics

The dataset contains **30 canonical Vietnamese sign classes**. All label text strings are Unicode NFC-normalized.

| Class ID | Canonical Label (NFC) | English Gloss | Total Videos | Canonical Videos |
|---|---|---|---|---|
| 0 | Ban ngày | Daytime | 144 | 104 |
| 1 | Ban đêm | Night | 128 | 88 |
| 2 | Bàn tay | Hand | 127 | 91 |
| 3 | Bạn thân | Best friend | 125 | 92 |
| 4 | Bệnh viện | Hospital | 121 | 89 |
| 5 | Chiều | Afternoon | 128 | 97 |
| 6 | Chào | Hello | 118 | 118 |
| 7 | Chân | Leg / Foot | 127 | 93 |
| 8 | Chúng ta | We / Us | 124 | 117 |
| 9 | Chậm lại | Slow down | 149 | 108 |
| 10 | Con gấu | Bear | 90 | 90 |
| 11 | Cá | Fish | 123 | 91 |
| 12 | Có thể | Can / Possible | 129 | 129 |
| 13 | Cơ thể | Body | 138 | 103 |
| 14 | Cứu | Save / Rescue | 128 | 96 |
| 15 | Dễ | Easy | 134 | 117 |
| 16 | Hôm nay | Today | 124 | 124 |
| 17 | Họ | They / Them | 117 | 117 |
| 18 | Học sinh | Student | 138 | 102 |
| 19 | Khóc | Cry | 150 | 118 |
| 20 | Mua | Buy | 133 | 99 |
| 21 | Mời vào | Welcome / Come in | 123 | 93 |
| 22 | Nghe | Listen / Hear | 132 | 100 |
| 23 | Ngón tay | Finger | 132 | 132 |
| 24 | Nhà | House / Home | 128 | 94 |
| 25 | Nhìn | Look / See | 126 | 126 |
| 26 | Nhầm | Mistake / Wrong | 132 | 100 |
| 27 | Nói | Speak / Talk | 129 | 100 |
| 28 | Nặng | Heavy | 132 | 99 |
| 29 | Ăn | Eat | 133 | 97 |
| **Total** | **30 Classes** | | **3,862** | **3,054** |

---

## 4. Video Properties and Quality Control

- **Total Clips:** 3,862 video files across 30 directories.
- **Canonical Clips:** 3,054 primary video recordings.
- **Derived Crop Variants:** 808 derived clips (pre-cut frame variants such as `cut_frames_35`, `cut_frames_40`).
- **Video Containers:** MP4 (70%) and AVI (30%).
- **Frame Rates:** 30 FPS and 60 FPS.
- **Resolutions:** 1280x720, 1920x1080, and 3840x2160 pixels.
- **Clip Durations:** ~110 to 368 frames (approximately 3 to 10 seconds per clip).
- **Signer Identities:** 522 unique signer camera sessions.

---

## 5. Pose Keypoint Extraction Format

Pose estimation features are extracted using **RTMPose Whole-body** (via `rtmlib` with `onnxruntime-gpu` acceleration).

### 5.1 Keypoint Layout (133 Joints)

Each frame produces a **133-keypoint whole-body layout**:
- **Body Joints:** 17 keypoints (indices 0..16)
- **Face Landmarks:** 68 keypoints (indices 17..84)
- **Left Hand:** 21 keypoints (indices 85..105)
- **Right Hand:** 21 keypoints (indices 106..126)
- **Foot Landmarks:** 6 keypoints (indices 127..132)

### 5.2 Keypoint File Structure (`.pkl`)

Extracted `.pkl` files stored under `data/CoSign/pose_format/` contain a pickled Python `dict`:

```python
{
    "keypoints": [  # List of numpy arrays per frame
        # Shape per frame: (1, 133, 2)
        # Normalized coordinates [x, y] in range [0.0, 1.0]
    ],
    "scores": [     # List of numpy arrays per frame
        # Shape per frame: (1, 133)
        # Confidence score per joint in range [0.0, 1.0]
    ]
}
```

---

## 6. Signer-Independent Split Protocol

To evaluate generalization to unseen signers, the dataset uses a **signer-independent split**:

- **Training Set (`train`):** 418 signers (2,509 canonical videos)
- **Development Set (`dev`):** 52 signers (274 canonical videos)
- **Test Set (`test`):** 52 signers (271 canonical videos)

**Key Split Guarantees:**
1. **Zero Signer Leakage:** No signer appears in more than one split.
2. **Full Class Coverage:** All 30 sign classes are represented in train, dev, and test sets.
3. **Deterministic Evaluation:** Evaluation on dev and test sets uses uniform temporal frame sampling to ensure 100% reproducible metrics.

---

## 7. Integration with Uni-Sign

### 7.1 Data Preparation Pipeline

```bash
# 1. Generate signer identity and split mappings
python script/build_signer_map.py \
  --data-root data/CoSign \
  --overwrite

# 2. Build Uni-Sign gzip-pickled split files and label manifests
python script/prepare_cosign.py \
  --data-root data/CoSign \
  --signer-map data/CoSign/metadata/signer_map.csv \
  --split-map data/CoSign/metadata/split_map.csv \
  --expected-label-count 30 \
  --allow-incomplete
```

### 7.2 Pose Extraction Command

```bash
# Extract RTMPose 133 keypoints using GPU ONNX Runtime
PYTHONPATH=demo/rtmlib-main python demo/pose_extraction.py \
  --src_dir data/CoSign \
  --tgt_dir data/CoSign/pose_format \
  --device cuda \
  --backend onnxruntime \
  --mode performance \
  --video_extensions mp4 avi \
  --recursive \
  --max_workers 1
```

### 7.3 Model Fine-Tuning Command

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
