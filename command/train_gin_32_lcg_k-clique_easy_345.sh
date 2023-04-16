#!/bin/bash
#SBATCH --job-name=train_gin_32_lcg_k-clique_easy_345
#SBATCH --output=/dev/null
#SBATCH --ntasks=1
#SBATCH --time=1-23:00:00
#SBATCH --gres=gpu:a100l:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16

module load anaconda/3
conda activate satbench

python train_model.py satisfiability ~/scratch/satbench/easy/k-clique/train/ \
    --train_splits sat unsat \
    --valid_dir ~/scratch/satbench/easy/k-clique/valid/ \
    --valid_splits sat unsat \
    --label satisfiability \
    --scheduler ReduceLROnPlateau \
    --lr 5e-05 \
    --n_iterations 32 \
    --weight_decay 1.e-8 \
    --model gin \
    --graph lcg \
    --seed 345
    --batch_size 64
