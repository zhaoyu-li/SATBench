#!/bin/bash
#SBATCH --job-name=eval_gin_32_lcg_3-sat2sr_easy_233
#SBATCH --output=%x_%j.out
#SBATCH --ntasks=1
#SBATCH --time=1-23:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16

module load anaconda/3
conda activate satbench

python eval_model.py satisfiability ~/scratch/satbench/easy/sr/test/ \
    ~/scratch/runs/task\=satisfiability_difficulty\=easy_dataset\=3-sat_splits\=sat_unsat_label\=satisfiability_loss\=None/graph=lcg_init_emb=learned_model=gin_n_iterations=32_seed=233_lr=5e-05_weight_decay=1e-08/checkpoints/model_best.pt \
    --model gin \
    --n_iterations 32 \
    --test_splits sat unsat \
    --label satisfiability \
