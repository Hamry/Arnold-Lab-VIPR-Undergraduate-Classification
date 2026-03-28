#!/bin/bash
#SBATCH --job-name=swin_final_v2
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=swin_final_v2_%j.out
#SBATCH --error=swin_final_v2_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mri28871@uga.edu

echo "SWIN FINAL V2 — 2ND OPTUNA BEST PARAMS"
cd $SLURM_SUBMIT_DIR
nvidia-smi
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/11.7.0
module load cuDNN/8.4.1.50-CUDA-11.7.0
pip install --user --quiet torch torchvision scikit-learn
python train_swin_final.py
echo "Done at $(date)"