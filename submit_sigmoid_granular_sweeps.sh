#!/bin/bash
# Submit all sigmoid granular layer sweep jobs to SLURM

BACKBONES=(
    alexnet
    convnext_base
    densenet201
    efficientnet_b4
    inception_v3
    resnet152
    swin_b
    vgg16
    vgg19
    vit_l_16
)

for BACKBONE in "${BACKBONES[@]}"; do
    CONFIG="configs/sigmoid_granular_layers/${BACKBONE}_sigmoid_granular_sweep.json"
    echo "Submitting: $CONFIG"
    sbatch run_optuna.slurm "$CONFIG"
done
