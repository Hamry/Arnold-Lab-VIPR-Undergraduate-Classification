#!/bin/bash

#SBATCH --job-name=inception_v3_opt
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=inception_v3_optimized_%j.out
#SBATCH --error=inception_v3_optimized_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mri28871@uga.edu

echo "======================================================================"
echo "OPTIMIZED INCEPTION V3 TRAINING - TARGET: 90%+ F1-SCORE"
echo "======================================================================"
echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "======================================================================"

# Navigate to your project directory
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

# Install required packages (only if not already installed)
echo ""
echo "Installing required packages..."
pip install --user --quiet torch torchvision scikit-learn matplotlib seaborn

# Run the optimized training script
echo ""
echo "======================================================================"
echo "Starting Optimized Inception v3 Training"
echo "======================================================================"
echo ""
python train_inception_v3_optimized.py

echo ""
echo "======================================================================"
echo "Job finished at $(date)"
echo "======================================================================"
