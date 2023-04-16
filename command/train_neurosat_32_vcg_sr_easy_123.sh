#!/bin/bash
#SBATCH --job-name=train_neurosat_32_vcg_sr_easy_123
#SBATCH --output=%x_%j.out
#SBATCH --ntasks=1
#SBATCH --time=1-23:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16

module load anaconda/3
conda activate satbench

python train_model.py satisfiability ~/scratch/satbench/easy/sr/train/ \
    --train_splits sat unsat \
    --valid_dir ~/scratch/satbench/easy/sr/valid/ \
    --valid_splits sat unsat \
    --label satisfiability \
    --scheduler ReduceLROnPlateau \
    --lr 0.0001 \
    --n_iterations 32 \
    --weight_decay 1.e-8 \
    --model neurosat \
    --graph vcg \
    --seed 123
