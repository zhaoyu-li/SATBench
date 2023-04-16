#!/bin/bash
#SBATCH --job-name=eval_ggnn_32_vcg_sr2sr_easy_233
#SBATCH --output=%x_%j.out
#SBATCH --ntasks=1
#SBATCH --time=1-23:00:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16

module load anaconda/3
conda activate satbench

python eval_model.py satisfiability ~/scratch/satbench/easy/sr/test/ \
    ~/scratch/runs/task\=satisfiability_difficulty\=easy_dataset\=sr_splits\=sat_unsat_label\=satisfiability_loss\=None/graph=vcg_init_emb=learned_model=ggnn_n_iterations=32_seed=233_lr=0.0001_weight_decay=1e-08/checkpoints/model_best.pt \
    --model ggnn \
    --graph vcg \
    --n_iterations 32 \
    --test_splits sat unsat \
    --label satisfiability \
