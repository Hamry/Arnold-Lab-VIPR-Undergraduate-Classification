#!/bin/bash

#SBATCH --job-name=vipr_batch
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=vipr_batch_%j.out
#SBATCH --error=vipr_batch_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hl98745@uga.edu

echo "======================================================================"
echo "VIPR CLASSIFICATION - BATCH BACKBONE BENCHMARK"
echo "======================================================================"
echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "======================================================================"

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

# Print GPU info
echo ""
echo "GPU Information:"
nvidia-smi
echo ""

# Load required modules
echo "Loading modules..."
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/11.7.0
module load cuDNN/8.4.1.50-CUDA-11.7.0

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source /scratch/hl98745/scratch_venv/bin/activate

# Verify Python and PyTorch
echo ""
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# Run all 7 models sequentially
echo "======================================================================"
echo "Starting batch training for all 7 backbone models"
echo "======================================================================"
echo ""

python run_batch.py \
    configs/resnet152.json \
    configs/densenet201.json \
    configs/vgg16.json \
    configs/alexnet.json \
    configs/efficientnet_b3.json \
    configs/inception_v3.json \
    configs/swin_t.json

echo ""
echo "======================================================================"
echo "Batch job finished at $(date)"
echo "======================================================================"
