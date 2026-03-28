#!/bin/bash
#SBATCH --job-name=inception_optuna
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=inception_optuna_%j.out
#SBATCH --error=inception_optuna_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mri28871@uga.edu

echo "======================================================================"
echo "INCEPTION V3 OPTUNA HYPERPARAMETER SEARCH"
echo "======================================================================"
echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "======================================================================"

cd $SLURM_SUBMIT_DIR

echo "GPU Information:"
nvidia-smi

echo "Loading modules..."
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/11.7.0
module load cuDNN/8.4.1.50-CUDA-11.7.0

echo "Installing packages..."
pip install --user --quiet torch torchvision scikit-learn matplotlib seaborn optuna

echo "Starting Optuna search..."
python train_inception_optuna.py

echo "======================================================================"
echo "Job finished at $(date)"
echo "======================================================================"