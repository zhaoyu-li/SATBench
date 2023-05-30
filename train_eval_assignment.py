import os
import subprocess
import numpy as np


def train():
    os.makedirs('command', exist_ok=True)
    for seed in [123, 234, 345]: # 123, 233, 345
        for dataset in ['sr', '3-sat', 'ca', 'ps', 'k-clique', 'k-domset']: # 'ca', 'ps', 'k-clique',  'k-domset', 'k-color'
            for model in ['gcn', 'gin', 'ggnn', 'neurosat']: # 'gcn', 'gin', 'ggnn', 'neurosat'
                for ite in [32]:
                    for graph in ['lcg', 'vcg']: # 'lcg', 'vcg'
                        for difficulty in ['medium']:
                            for loss in ['supervised', 'unsupervised', 'unsupervisedv2']:
                                if model == 'neurosat' and graph == 'vcg':
                                    continue

                                file_name = f'command/assignment_train_{model}_{ite}_{graph}_{dataset}_{difficulty}_{loss}_{seed}.sh'
                                with open(file_name, 'w') as f:
                                    # if model == 'gin':
                                    #    lr = 5.e-5
                                    # else:
                                    #    lr = 1.e-4
                                    lr = 1.e-4
                                    if dataset in ['k-domset', 'k-clique'] and difficulty == 'medium':
                                        batch_size = 64
                                    else:
                                        batch_size = 128
                                    # batch_size = 128
                                    f.write('#!/bin/bash\n')
                                    f.write(f'#SBATCH --job-name=assignment_train_{model}_{ite}_{graph}_{dataset}_{difficulty}_{loss}_{seed}\n')
                                    f.write('#SBATCH --output=/dev/null\n')
                                    f.write('#SBATCH --ntasks=1\n')
                                    f.write('#SBATCH --time=2-12:00:00\n')
                                    f.write('#SBATCH --gres=gpu:rtx8000:1\n')
                                    f.write('#SBATCH --mem=16G\n')
                                    f.write('#SBATCH --cpus-per-task=8\n')
                                    f.write('\n')
                                    f.write('module load anaconda/3\n')
                                    f.write('conda activate satbench\n')
                                    f.write('\n')
                                    f.write(f'python train_model.py assignment $SCRATCH/satbench/{difficulty}/{dataset}/train/ \\\n')
                                    f.write('    --train_splits sat \\\n')
                                    f.write(f'    --valid_dir $SCRATCH/satbench/{difficulty}/{dataset}/valid/ \\\n')
                                    f.write('    --valid_splits sat \\\n')
                                    if loss == 'supervised':
                                        f.write('    --label assignment \\\n')
                                    f.write(f'    --loss {loss} \\\n')
                                    f.write('    --scheduler ReduceLROnPlateau \\\n')
                                    f.write(f'    --lr {lr} \\\n')
                                    f.write(f'    --n_iterations {ite} \\\n')
                                    f.write('    --weight_decay 1.e-8 \\\n')
                                    f.write(f'    --model {model} \\\n')
                                    f.write(f'    --graph {graph} \\\n')
                                    f.write(f'    --seed {seed} \\\n')
                                    f.write(f'    --batch_size {batch_size}\n')
                                result = subprocess.run(
                                    ['sbatch', file_name],
                                    capture_output=False, text=False)
                                if result.returncode == 0:
                                    print("Job submitted successfully.")
                                else:
                                    print(f"Job submission failed with error: {result.stderr}")

def eval():
    file_name = f'command/assignment_eval.sh'
    with open(file_name, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write(
            f'#SBATCH --job-name=eval\n')
        f.write('#SBATCH --output=/dev/null\n')
        f.write('#SBATCH --ntasks=1\n')
        f.write('#SBATCH --time=1-23:00:00\n')
        f.write('#SBATCH --gres=gpu:rtx8000:1\n')
        f.write('#SBATCH --mem=16G\n')
        f.write('#SBATCH --cpus-per-task=16\n')
        f.write('\n')
        f.write('module load anaconda/3\n')
        f.write('conda activate satbench\n')
        f.write('\n')
        os.makedirs('command', exist_ok=True)
        for seed in [123, 234, 345]:
            for dataset in ['sr', '3-sat', 'ca', 'ps', 'k-clique', 'k-domset']:
                for model in ['gcn', 'gin', 'ggnn', 'neurosat']:
                    for ite in [32]:
                        for graph in ['lcg', 'vcg']:
                            for difficulty in ['easy']:
                                for loss in ['unsupervised', 'unsupervisedv2', 'supervised']:
                                    if model == 'neurosat' and graph == 'vcg':
                                        continue
                                    if loss == 'supervised':
                                        label = 'assignment'
                                    else:
                                        label = 'None'
                                    lr = '0.0001'
                                    if dataset in ['k-domset', 'k-clique'] and difficulty == 'medium':
                                        batch_size = 64
                                    else:
                                        batch_size = 128

                                    f.write(
                                        f'python eval_model.py assignment /network/scratch/x/xujie.si/satbench/{difficulty}/{dataset}/test/ \\\n')
                                    f.write(f'    /network/scratch/x/xujie.si/runs/task\=assignment_difficulty\={difficulty}_dataset\={dataset}_splits\=sat_label\={label}_loss\={loss}/graph={graph}_init_emb=learned_model={model}_n_iterations={ite}_seed={seed}_lr={lr}_weight_decay=1e-08/checkpoints/model_best.pt \\\n')
                                    f.write(f'    --model {model} \\\n')
                                    f.write(f'    --graph {graph} \\\n')
                                    f.write(f'    --n_iterations {ite} \\\n')
                                    f.write(f'    --batch_size {batch_size} \\\n')
                                    f.write(f'    --test_splits sat\n')

                                    f.write(f'\n')

    result = subprocess.run(
        ['sbatch', f'command/assignment_eval.sh'],
        capture_output=False, text=False)
    if result.returncode == 0:
        print("Job submitted successfully.")
    else:
        print(f"Job submission failed with error: {result.stderr}")


def summary():
    acc_dict = {}
    for loss in ['unsupervised', 'unsupervisedv2', 'supervised']:
        acc_dict[f'{loss}'] = {}
        for dataset in ['sr', '3-sat', 'ca', 'ps', 'k-clique', 'k-domset']:
            acc_dict[f'{loss}'][f'{dataset}'] = {}
            for model in ['gcn', 'gin', 'ggnn', 'neurosat']:
                for ite in [32]:
                    for graph in ['lcg', 'vcg']:
                        if model == 'neurosat' and graph == 'vcg':
                            continue
                        acc_dict[f'{loss}'][f'{dataset}'][f'{model}_{graph}'] = {}
                        for difficulty in ['easy']:
                            acc = []
                            for seed in [123, 234, 345]:
                                if loss == 'supervised':
                                    label = 'assignment'
                                else:
                                    label = 'None'
                                lr = '0.0001'
                                dir = f'/network/scratch/x/xujie.si/runs/task=assignment_difficulty={difficulty}_dataset={dataset}_splits=sat_label={label}_loss={loss}/graph={graph}_init_emb=learned_model={model}_n_iterations={ite}_seed={seed}_lr={lr}_weight_decay=1e-08/'
                                file_name = f'eval_task=assignment_difficulty={difficulty}_dataset={dataset}_splits=sat_decoding=standard_n_iterations={ite}_checkpoint=model_best.txt'
                                with open(os.path.join(dir, file_name), 'r') as ff:
                                    lines = ff.readlines()

                                for line in lines:
                                    if 'Testing accuracy' in line:
                                        line = line.replace(' ', '').split(':')
                                        acc.append(float(line[1]))
                                        break
                            # print(dataset, model, graph, loss, acc)
                            assert len(acc) == 3
                            acc = np.array(acc)
                            mean = np.mean(acc)
                            std = np.std(acc)

                            acc_dict[f'{loss}'][f'{dataset}'][f'{model}_{graph}']['mean'] = mean
                            acc_dict[f'{loss}'][f'{dataset}'][f'{model}_{graph}']['std'] = std

    os.makedirs('results', exist_ok=True)
    file_name = f'results/assignment_eval.csv'

    with open(file_name, 'w') as f:
        for loss in ['unsupervised', 'unsupervisedv2', 'supervised']:
            print(f'{loss}', file=f)
            print('mean', file=f, end='')
            for model in ['gcn', 'gin', 'ggnn', 'neurosat']:
                for graph in ['lcg', 'vcg']:
                    if model == 'neurosat' and graph == 'vcg':
                        continue
                    print(f',{model}_{graph}', file=f, end='')
            print('\n', file=f, end='')
            for dataset in ['sr', '3-sat', 'ca', 'ps', 'k-clique', 'k-domset']:
                print(f'{dataset}', file=f, end='')
                for model in ['gcn', 'gin', 'ggnn', 'neurosat']:
                    for graph in ['lcg', 'vcg']:
                        if model == 'neurosat' and graph == 'vcg':
                            continue
                        print(f',{acc_dict[f"{loss}"][f"{dataset}"][f"{model}_{graph}"]["mean"]}', file=f, end='')
                print('\n', file=f, end='')

            print(file=f)
            print('std', file=f, end='')
            for model in ['gcn', 'gin', 'ggnn', 'neurosat']:
                for graph in ['lcg', 'vcg']:
                    if model == 'neurosat' and graph == 'vcg':
                        continue
                    print(f',{model}_{graph}', file=f, end='')
            print('\n', file=f, end='')
            for dataset in ['sr', '3-sat', 'ca', 'ps', 'k-clique', 'k-domset']:
                print(f'{dataset}', file=f, end='')
                for model in ['gcn', 'gin', 'ggnn', 'neurosat']:
                    for graph in ['lcg', 'vcg']:
                        if model == 'neurosat' and graph == 'vcg':
                            continue
                        print(f',{acc_dict[f"{loss}"][f"{dataset}"][f"{model}_{graph}"]["std"]}', file=f, end='')
                print('\n', file=f, end='')
            print('\n', file=f, end='')

# summary()
train()
