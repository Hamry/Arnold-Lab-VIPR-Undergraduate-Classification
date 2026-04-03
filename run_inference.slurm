#!/bin/bash

#SBATCH --job-name=vipr_inference
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=vipr_inference_%j.out
#SBATCH --error=vipr_inference_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hl98745@uga.edu

echo "======================================================================"
echo "VIPR CLASSIFICATION - INFERENCE"
echo "======================================================================"
echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "======================================================================"

cd $SLURM_SUBMIT_DIR

echo ""
echo "GPU Information:"
nvidia-smi
echo ""

echo "Loading modules..."
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/11.7.0
module load cuDNN/8.4.1.50-CUDA-11.7.0

echo ""
echo "Activating virtual environment..."
source /scratch/hl98745/scratch_venv/bin/activate

echo ""
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# Arguments (override via: sbatch run_inference.slurm <result-dir> <data-dir> [output-dir])
RESULT_DIR="${1:?ERROR: result-dir argument is required. Usage: sbatch run_inference.slurm <result-dir> <data-dir> [output-dir]}"
DATA_DIR="${2:?ERROR: data-dir argument is required. Usage: sbatch run_inference.slurm <result-dir> <data-dir> [output-dir]}"
OUTPUT_DIR="${3:-}"

echo "======================================================================"
echo "Result dir : $RESULT_DIR"
echo "Data dir   : $DATA_DIR"
echo "Output dir : ${OUTPUT_DIR:-<result-dir>/inference_output}"
echo "======================================================================"
echo ""

if [ -n "$OUTPUT_DIR" ]; then
    python run_inference.py \
        --result-dir "$RESULT_DIR" \
        --data "$DATA_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --workers 8
else
    python run_inference.py \
        --result-dir "$RESULT_DIR" \
        --data "$DATA_DIR" \
        --workers 8
fi

echo ""
echo "======================================================================"
echo "Inference job finished at $(date)"
echo "======================================================================"
