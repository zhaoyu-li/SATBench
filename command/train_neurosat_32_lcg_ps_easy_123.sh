#!/bin/bash
#SBATCH --job-name=train_neurosat_32_lcg_ps_easy_123
#SBATCH --output=/dev/null
#SBATCH --ntasks=1
#SBATCH --time=1-23:00:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16

module load anaconda/3
conda activate satbench

python train_model.py satisfiability $SCRATCH/satbench/easy/ps/train/ \
    --train_splits sat unsat \
    --valid_dir $SCRATCH/satbench/easy/ps/valid/ \
    --valid_splits sat unsat \
    --label satisfiability \
    --scheduler ReduceLROnPlateau \
    --lr 0.0001 \
    --n_iterations 32 \
    --weight_decay 1.e-8 \
    --model neurosat \
    --graph lcg \
    --seed 123
    --batch_size 128
