#!/bin/bash
#SBATCH --job-name=train_gin_32_vcg_ca_easy_345
#SBATCH --output=/dev/null
#SBATCH --ntasks=1
#SBATCH --time=1-23:00:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16

module load anaconda/3
conda activate satbench

python train_model.py satisfiability $SCRATCH/satbench/easy/ca/train/ \
    --train_splits sat unsat \
    --valid_dir $SCRATCH/satbench/easy/ca/valid/ \
    --valid_splits sat unsat \
    --label satisfiability \
    --scheduler ReduceLROnPlateau \
    --lr 5e-05 \
    --n_iterations 32 \
    --weight_decay 1.e-8 \
    --model gin \
    --graph vcg \
    --seed 345 \
    --batch_size 128
