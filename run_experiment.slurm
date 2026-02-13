#!/bin/bash

#SBATCH --job-name=vipr_train
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=vipr_train_%j.out
#SBATCH --error=vipr_train_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hl98745@uga.edu

echo "======================================================================"
echo "VIPR CLASSIFICATION - BACKBONE BENCHMARK"
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

# Run the training script
# Pass config file as argument, defaults to resnet152_baseline.json
CONFIG_FILE="${1:-configs/resnet152.json}"

echo "======================================================================"
echo "Starting training with config: $CONFIG_FILE"
echo "======================================================================"
echo ""

python run_experiment.py "$CONFIG_FILE"

echo ""
echo "======================================================================"
echo "Job finished at $(date)"
echo "======================================================================"
