#!/bin/bash
#SBATCH --job-name=swin_optuna
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=swin_optuna_%j.out
#SBATCH --error=swin_optuna_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mri28871@uga.edu

echo "======================================================================"
echo "SWIN TRANSFORMER OPTUNA HYPERPARAMETER SEARCH"
echo "======================================================================"
echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"

cd $SLURM_SUBMIT_DIR

nvidia-smi

module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/11.7.0
module load cuDNN/8.4.1.50-CUDA-11.7.0

pip install --user --quiet torch torchvision scikit-learn matplotlib seaborn optuna

python train_swin_optuna.py

echo "Job finished at $(date)"