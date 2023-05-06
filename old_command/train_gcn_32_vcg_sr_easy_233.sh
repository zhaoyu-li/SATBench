#!/bin/bash
#SBATCH --job-name=train_gcn_32_vcg_sr_easy_233
#SBATCH --output=/dev/null
#SBATCH --ntasks=1
#SBATCH --time=5-23:00:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

module load anaconda/3
conda activate satbench

python train_model.py assignment $SCRATCH/satbench/easy/sr/train/ \
    --train_splits sat \
    --valid_dir $SCRATCH/satbench/easy/sr/valid/ \
    --valid_splits sat \
    --label assignment \
    --loss supervised \
    --scheduler ReduceLROnPlateau \
    --lr 5e-05 \
    --n_iterations 32 \
    --weight_decay 1.e-8 \
    --model gcn \
    --graph vcg \
    --seed 233 \
    --batch_size 128
