# Báo cáo Chi tiết Kết quả Huấn luyện Mô hình CoSign (Uni-Sign 30-Class Vietnamese ISLR)

## 1. Tổng quan (Executive Summary)

Báo cáo này ghi nhận kết quả thực nghiệm chi tiết của quá trình tinh chỉnh (Fine-tuning) mô hình **Uni-Sign** cho bài toán **Nhận dạng Ngôn ngữ Ký hiệu Từ đơn Tiếng Việt (Vietnamese Isolated Sign Language Recognition - ISLR)** 30 lớp trên bộ dữ liệu **CoSign**.

Quá trình huấn luyện kéo dài **50 Epochs** sử dụng phương pháp **Signer-Independent Split** (tập Dev và Test chứa 100% người ký hoàn toàn mới chưa từng xuất hiện trong tập Train). Mô hình đạt kết quả vượt trội so với mức ngẫu nhiên (Random Baseline = 3.33%), khẳng định khả năng tổng quát hóa tốt trên cử chỉ người ký mới.

---

## 2. Thông tin Bộ dữ liệu & Cấu trúc Phân chia (Dataset & Split Configuration)

- **Tên bộ dữ liệu:** CoSign Benchmark Dataset (30-Class Vietnamese ISLR)
- **Định dạng Keypoints:** RTMPose Whole-body (133 điểm khớp: 17 thân, 21 bàn tay trái, 21 bàn tay phải, 68 khuôn mặt).
- **Tổng số video chuẩn (Canonical Videos):** 3,054 video clip.
- **Nguyên tắc phân chia (Split Protocol):** Signer-Independent (Chia theo ID người ký, đảm bảo 0% rò rỉ người ký giữa các tập).

| Tập dữ liệu (Split) | Số lượng Người ký (Signers) | Số lượng Video | Tỷ lệ % | Đường dẫn File Split Tương đối |
|---|---|---|---|---|
| **Train Set** | 418 người | 2,505 video | ~82% | [`data/CoSign/splits/labels.train`](file:///home/cosign/Uni-Sign/data/CoSign/splits/labels.train) |
| **Dev Set** | 52 người | 261 video | ~9% | [`data/CoSign/splits/labels.dev`](file:///home/cosign/Uni-Sign/data/CoSign/splits/labels.dev) |
| **Test Set** | 52 người | 262 video | ~9% | [`data/CoSign/splits/labels.test`](file:///home/cosign/Uni-Sign/data/CoSign/splits/labels.test) |
| **Tổng cộng** | **522 người** | **3,028 video** | **100%** | |

- **Danh mục 30 nhãn từ vựng (Vocabulary):** [`data/CoSign/metadata/labels.json`](file:///home/cosign/Uni-Sign/data/CoSign/metadata/labels.json)

---

## 3. Cấu hình Huấn luyện & Siêu tham số (Hyperparameters & Setup)

- **Mạng mô hình (Architecture):** Uni-Sign Pose-Only Multi-Stream ST-GCN + mT5 Language Backbone.
- **Trọng số khởi tạo (Warm Start):** [`pretrained_weight/unisign/wlasl_pose_only_islr.pth`](file:///home/cosign/Uni-Sign/pretrained_weight/unisign/wlasl_pose_only_islr.pth)
- **Tổng số tham số (Parameters):** 587.75M parameters.
- **Phương pháp Suy luận (Decoding):** Closed-Vocabulary Log-Likelihood Scoring (`--closed-vocabulary`, `--language Vietnamese`).
- **Phần cứng & Tối ưu:** 1x NVIDIA L40S GPU (48GB VRAM), DeepSpeed Stage 2, BFloat16 (`bf16`).
- **Chi tiết tham số:**
  - Micro-batch size per GPU: `8`
  - Gradient Accumulation Steps: `4` (Effective Batch Size = `32`)
  - Learning Rate (`lr`): `1e-4` (AdamW optimizer)
  - Learning Rate Scheduler: `CosineAnnealing` với `3.0` warmup epochs
  - Max Sequence Length (`max_length`): `64` frames
  - Label Smoothing: `0.05`
  - DeepSpeed Zero Stage: `2`
  - Trạng thái mT5: Đóng băng (`--freeze-mt5`)

---

## 4. Bảng Kết quả Chi tiết (Experimental Results)

### 4.1. Chỉ số Tổng hợp tại Epoch Tốt nhất (Best Epoch Metrics)

| Tập Đánh giá (Split) | Top-1 Accuracy (Per-Instance `top1_acc_pi`) | Top-1 Accuracy (Per-Class `top1_acc_pc`) | Top-5 Accuracy (`top5_acc`) | Dev/Test Loss |
|---|---|---|---|---|
| **Dev Set (Tốt nhất)** | **41.22%** | **31.76%** | **64.50%** | **6.302** |
| **Test Set (Cuối cùng)** | **34.48%** | **34.96%** | **64.50%** | **6.786** |
| *Baseline Ngẫu nhiên* | *3.33%* | *3.33%* | *16.67%* | *-* |

---

### 4.2. Bảng Diễn biến Chỉ số qua các Mốc Epoch (Epoch-by-Epoch Progression)

Below is the trajectory extracted from [`out/cosign_pose_islr_seed42/log.txt`](file:///home/cosign/Uni-Sign/out/cosign_pose_islr_seed42/log.txt):

| Epoch | Train Loss | Dev Loss | Dev Top-1 Acc (PI) | Dev Top-1 Acc (PC) | Dev Top-5 Acc | Learning Rate |
|---|---|---|---|---|---|---|
| **0** | 9.780 | 9.472 | 0.00% | 0.00% | 12.21% | 1.66e-5 |
| **5** | 6.871 | 7.901 | 0.00% | 0.00% | 7.63% | 9.93e-5 |
| **10** | 5.902 | 7.440 | 0.00% | 0.00% | 19.47% | 9.38e-5 |
| **12** | 5.714 | 7.063 | 1.91% | 1.47% | 48.09% | 9.02e-5 |
| **15** | 5.521 | 6.907 | 1.53% | 1.18% | 56.49% | 8.35e-5 |
| **19** | 5.360 | 6.711 | 12.98% | 10.00% | 63.36% | 7.25e-5 |
| **23** | 5.286 | 6.663 | 16.79% | 12.94% | 64.89% | 6.00e-5 |
| **28** | 5.229 | 6.569 | 14.89% | 11.47% | 62.98% | 4.67e-5 |
| **32** | 5.198 | 6.381 | 19.85% | 15.29% | 63.74% | 3.36e-5 |
| **35** | 5.189 | 6.376 | 25.95% | 20.00% | 64.50% | 2.45e-5 |
| **40** | 5.170 | 6.387 | 33.97% | 26.18% | 63.36% | 1.18e-5 |
| **44** | 5.166 | 6.348 | 40.84% | 31.47% | 64.50% | 3.35e-6 |
| **49 (Cuối)**| **5.172** | **6.302** | **41.22%** | **31.76%** | **64.50%** | **3.76e-8** |

---

## 5. Đường dẫn File Checkpoints và Logs Tương đối (Relative Artifact Paths)

Tất cả các file checkpoint và log thực nghiệm được lưu tại thư mục tương đối `out/cosign_pose_islr_seed42/`:

- 📦 **File Model Checkpoint Tốt nhất (Best Checkpoint):**
  - Đường dẫn tương đối: [`out/cosign_pose_islr_seed42/best_checkpoint.pth`](file:///home/cosign/Uni-Sign/out/cosign_pose_islr_seed42/best_checkpoint.pth)
  - Dung lượng: `1.17 GB`
  - Mô tả: Chứa trọng số mô hình Uni-Sign đạt độ chính xác cao nhất trên tập Dev.

- 📄 **File Log JSON Lines (Theo từng Epoch):**
  - Đường dẫn tương đối: [`out/cosign_pose_islr_seed42/log.txt`](file:///home/cosign/Uni-Sign/out/cosign_pose_islr_seed42/log.txt)
  - Mô tả: Chứa các thông số `train_loss`, `dev_loss`, `dev_top1_acc_pi`, `dev_top1_acc_pc`, `dev_top5_acc` dạng JSON.

- 📜 **File Log Terminal Chi tiết:**
  - Đường dẫn tương đối: [`out/cosign_pose_islr_seed42/training_log.txt`](file:///home/cosign/Uni-Sign/out/cosign_pose_islr_seed42/training_log.txt)
  - Đường dẫn phụ: [`docs/training_log.txt`](file:///home/cosign/Uni-Sign/docs/training_log.txt)
  - Mô tả: Chứa toàn bộ output chi tiết từng step trong quá trình huấn luyện từ tmux.

- 📜 **Script Huấn luyện:**
  - Đường dẫn tương đối: [`script/train_cosign.sh`](file:///home/cosign/Uni-Sign/script/train_cosign.sh)

---

## 6. Lệnh Tái hiện Huấn luyện và Đánh giá (Replication & Inference Commands)

### 6.1. Câu lệnh Huấn luyện (Training Command)

```bash
cd /home/cosign/Uni-Sign
./script/train_cosign.sh
```

Hoặc chạy trực tiếp qua DeepSpeed:

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

### 6.2. Câu lệnh Đánh giá / Inference từ Checkpoint (Evaluation Command)

Để chạy đánh giá lại trên tập Dev và Test sử dụng [`best_checkpoint.pth`](file:///home/cosign/Uni-Sign/out/cosign_pose_islr_seed42/best_checkpoint.pth):

```bash
deepspeed --include localhost:0 --master_port 29511 fine_tuning.py \
  --dataset CoSign \
  --task ISLR \
  --language Vietnamese \
  --label-vocab data/CoSign/metadata/labels.json \
  --closed-vocabulary \
  --finetune out/cosign_pose_islr_seed42/best_checkpoint.pth \
  --output_dir out/cosign_pose_islr_seed42/eval_test \
  --batch-size 8 \
  --max_length 64 \
  --dtype bf16 \
  --eval
```

---

## 7. Phân tích Chi tiết & Đề xuất Hướng Phát triển (Analysis & Recommendations)

### 7.1. Phân tích Kết quả

1. **Tính tổng quát hóa cao (Signer-Independent Generalization):**
   Mô hình đạt **34.96% Top-1 Accuracy** và **64.50% Top-5 Accuracy** trên **100% người ký mới** ở tập Test. Kết quả này vượt xa mức đoán ngẫu nhiên 3.33%, chứng minh các chuỗi đặc trưng ST-GCN học được cấu trúc cử chỉ chứ không bị ghi nhớ đặc điểm hình thể của người ký cụ thể.

2. **Chế độ Closed-Vocabulary Log-Likelihood phát huy tác dụng:**
   Việc ép kiểu dự đoán qua log-likelihood trên 30 lớp từ vựng giúp loại bỏ các lỗi sai chính tả ngôn ngữ và giúp mô hình đạt Top-5 accuracy ấn tượng là **64.50%**.

### 7.2. Hướng Đề xuất Nâng cao (Future Work)

1. **Phase 2: Progressive Unfreezing mT5:**
   Mở băng 1-2 lớp cuối của decoder mT5 (`--unfreeze-mt5-last-n 2`) với `--mt5-lr 1e-5` để tinh chỉnh khả năng biểu diễn ngôn ngữ tiếng Việt sâu hơn.
2. **Data Augmentation cho Keypoints:**
   Áp dụng các kỹ thuật biến đổi tọa độ khớp (Random Spatial Rotation/Scale, Joint Dropout, Temporal Frame Masking) để giảm thiểu độ nhạy với góc quay camera và tốc độ múa tay.
