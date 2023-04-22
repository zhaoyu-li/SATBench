import os
import subprocess


def train():
    os.makedirs('command', exist_ok=True)
    for seed in [233, 345, 123]: # 123, 233, 345
        for dataset in ['k-domset']: # 'ca', 'ps', 'k-clique',  'k-domset', 'k-color'
            for model in ['gcn', 'gin', 'ggnn', 'neurosat']: # 'gcn', 'gin', 'ggnn', 'neurosat'
                for ite in [32]:
                    for graph in ['lcg', 'vcg']: # 'lcg', 'vcg'
                        for difficulty in ['easy']:
                            if model == 'neurosat' and graph == 'vcg':
                                continue
                            with open(f'command/train_{model}_{ite}_{graph}_{dataset}_{difficulty}_{seed}.sh', 'w') as f:
                                if model == 'gin':
                                    lr = 5.e-5
                                else:
                                    lr = 1.e-4
                                if dataset in ['k-domset', 'k-clique']:
                                    batch_size = 64
                                else:
                                    batch_size = 128
                                f.write('#!/bin/bash\n')
                                f.write(f'#SBATCH --job-name=train_{model}_{ite}_{graph}_{dataset}_{difficulty}_{seed}\n')
                                f.write('#SBATCH --output=/dev/null\n')
                                f.write('#SBATCH --ntasks=1\n')
                                f.write('#SBATCH --time=5-23:00:00\n')
                                f.write('#SBATCH --gres=gpu:rtx8000:1\n')
                                f.write('#SBATCH --mem=32G\n')
                                f.write('#SBATCH --cpus-per-task=16\n')
                                f.write('\n')
                                f.write('module load anaconda/3\n')
                                f.write('conda activate satbench\n')
                                f.write('\n')
                                f.write(f'python train_model.py satisfiability $SCRATCH/satbench/{difficulty}/{dataset}/train/ \\\n')
                                f.write('    --train_splits sat unsat \\\n')
                                f.write(f'    --valid_dir $SCRATCH/satbench/{difficulty}/{dataset}/valid/ \\\n')
                                f.write('    --valid_splits sat unsat \\\n')
                                f.write('    --label satisfiability \\\n')
                                f.write('    --scheduler ReduceLROnPlateau \\\n')
                                f.write(f'    --lr {lr} \\\n')
                                f.write(f'    --n_iterations {ite} \\\n')
                                f.write('    --weight_decay 1.e-8 \\\n')
                                f.write(f'    --model {model} \\\n')
                                f.write(f'    --graph {graph} \\\n')
                                f.write(f'    --seed {seed} \\\n')
                                f.write(f'    --batch_size {batch_size}\n')
                            result = subprocess.run(
                                ['sbatch', f'command/train_{model}_{ite}_{graph}_{dataset}_{difficulty}_{seed}.sh'],
                                capture_output=False, text=False)
                            if result.returncode == 0:
                                print("Job submitted successfully.")
                            else:
                                print(f"Job submission failed with error: {result.stderr}")


def eval():
    os.makedirs('command', exist_ok=True)
    for seed in [123, 233, 345]:
        for ori_dataset in ['3-sat', 'sr']:
            for target_dataset in ['3-sat', 'sr']:
                for model in ['gcn', 'gin', 'ggnn', 'neurosat']:
                    for ite in [32]:
                        for graph in ['lcg', 'vcg']:
                            for difficulty in ['easy']:
                                if model == 'neurosat' and graph == 'vcg':
                                    continue
                                with open(f'command/eval_{model}_{ite}_{graph}_ori_{ori_dataset}_target_{target_dataset}_{difficulty}_{seed}.sh',
                                          'w') as f:
                                    if model == 'gin':
                                        lr = '5e-05'
                                    else:
                                        lr = '0.0001'
                                    if dataset in ['k-domset', 'k-clique']:
                                        batch_size = 64
                                    else:
                                        batch_size = 128
                                    f.write('#!/bin/bash\n')
                                    f.write(
                                        f'#SBATCH --job-name=eval_{model}_{ite}_{graph}_{ori_dataset}2{target_dataset}_{difficulty}_{seed}\n')
                                    f.write('#SBATCH --output=/dev/null\n')
                                    f.write('#SBATCH --ntasks=1\n')
                                    f.write('#SBATCH --time=1-23:00:00\n')
                                    f.write('#SBATCH --gres=gpu:rtx8000:1\n')
                                    f.write('#SBATCH --mem=32G\n')
                                    f.write('#SBATCH --cpus-per-task=16\n')
                                    f.write('\n')
                                    f.write('module load anaconda/3\n')
                                    f.write('conda activate satbench\n')
                                    f.write('\n')
                                    f.write(
                                        f'python eval_model.py satisfiability $SCRATCH/satbench/{difficulty}/{target_dataset}/test/ \\\n')
                                    f.write(f'    $SCRATCH/runs/task\=satisfiability_difficulty\={difficulty}_dataset\={ori_dataset}_splits\=sat_unsat_label\=satisfiability_loss\=None/graph={graph}_init_emb=learned_model={model}_n_iterations={ite}_seed={seed}_lr={lr}_weight_decay=1e-08/checkpoints/model_best.pt \\\n')
                                    f.write(f'    --model {model} \\\n')
                                    f.write(f'    --graph {graph} \\\n')
                                    f.write(f'    --n_iterations {ite} \\\n')
                                    f.write(f'    --batch_size {batch_size}\\\n')
                                    f.write(f'    --test_splits sat unsat \\\n')
                                    f.write(f'    --label satisfiability \\\n')

                                result = subprocess.run(
                                    ['sbatch', f'command/eval_{model}_{ite}_{graph}_ori_{ori_dataset}_target_{target_dataset}_{difficulty}_{seed}.sh'],
                                    capture_output=False, text=False)
                                if result.returncode == 0:
                                    print("Job submitted successfully.")
                                else:
                                    print(f"Job submission failed with error: {result.stderr}")
train()
