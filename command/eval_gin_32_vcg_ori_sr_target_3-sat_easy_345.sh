#!/bin/bash
#SBATCH --job-name=eval_gin_32_vcg_sr23-sat_easy_345
#SBATCH --output=%x_%j.out
#SBATCH --ntasks=1
#SBATCH --time=1-23:00:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16

module load anaconda/3
conda activate satbench

python eval_model.py satisfiability ~/scratch/satbench/easy/3-sat/test/ \
    ~/scratch/runs/task\=satisfiability_difficulty\=easy_dataset\=sr_splits\=sat_unsat_label\=satisfiability_loss\=None/graph=vcg_init_emb=learned_model=gin_n_iterations=32_seed=345_lr=5e-05_weight_decay=1e-08/checkpoints/model_best.pt \
    --model gin \
    --graph vcg \
    --n_iterations 32 \
    --test_splits sat unsat \
    --label satisfiability \
