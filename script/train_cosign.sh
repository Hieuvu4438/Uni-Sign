#!/usr/bin/env bash
set -e

source /home/cosign/miniconda3/etc/profile.d/conda.sh
conda activate unisign

# Path setup
OUTPUT_DIR="out/cosign_pose_islr_seed42"
LABEL_VOCAB="data/CoSign/metadata/labels.json"
PRETRAINED_CKPT="pretrained_weight/unisign/wlasl_pose_only_islr.pth"

# GPU selection
GPUS="localhost:0"
PORT="29511"

echo "=== Starting Uni-Sign Fine-Tuning on CoSign Dataset (30-class Vietnamese ISLR) ==="
echo "Output Directory: ${OUTPUT_DIR}"
echo "Pretrained Checkpoint: ${PRETRAINED_CKPT}"

mkdir -p "${OUTPUT_DIR}"

deepspeed --include "${GPUS}" --master_port "${PORT}" fine_tuning.py \
  --dataset CoSign \
  --task ISLR \
  --language Vietnamese \
  --label-vocab "${LABEL_VOCAB}" \
  --closed-vocabulary \
  --freeze-mt5 \
  --finetune "${PRETRAINED_CKPT}" \
  --output_dir "${OUTPUT_DIR}" \
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

echo "=== Training Complete! Best checkpoint saved to ${OUTPUT_DIR}/best_checkpoint.pth ==="
