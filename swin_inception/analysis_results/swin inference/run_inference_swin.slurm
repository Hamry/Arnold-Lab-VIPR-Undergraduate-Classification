#!/bin/bash
#SBATCH --job-name=infer_swin
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=infer_swin_%j.out
#SBATCH --error=infer_swin_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mri28871@uga.edu

# ── Usage ─────────────────────────────────────────────────────────────────────
# Basic (final model weights, no config override):
#   sbatch run_inference_swin.slurm
#
# With explicit paths:
#   sbatch run_inference_swin.slurm \
#       /path/to/best_model.pth \
#       /work/omlpa/SAM \
#       predictions_swin.csv
#
# Positional args: $1=pth  $2=data_dir  $3=out_csv  $4=config_json (optional)
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PTH="${1:-results_swin_final/best_model.pth}"
DATA_DIR="${2:-/work/omlpa/SAM}"
OUT_CSV="${3:-predictions_swin.csv}"
CONFIG="${4:-}"          # optional: pass optuna_best_params.json as 4th arg

echo "======================================================================"
echo "SWIN TRANSFORMER INFERENCE — MycorrhiSEE Dataset"
echo "======================================================================"
echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "  Model  : $MODEL_PTH"
echo "  Data   : $DATA_DIR"
echo "  Output : $OUT_CSV"
echo "======================================================================"

cd $SLURM_SUBMIT_DIR
nvidia-smi

module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/11.7.0
module load cuDNN/8.4.1.50-CUDA-11.7.0

pip install --user --quiet torch torchvision pillow

echo ""
echo "Starting inference..."
echo ""

CMD="python inference_swin.py \
    --model  \"$MODEL_PTH\" \
    --data   \"$DATA_DIR\" \
    --output \"$OUT_CSV\" \
    --batch-size 32"

[ -n "$CONFIG" ] && CMD="$CMD --config \"$CONFIG\""
# Uncomment to enable TTA:
# CMD="$CMD --tta"

eval $CMD

echo ""
echo "======================================================================"
echo "Job finished at $(date)"
echo "Output written to: $OUT_CSV"
echo "======================================================================"
