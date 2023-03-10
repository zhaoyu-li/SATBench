import torch
import torch.nn.functional as F
import os
import sys
import argparse
import pickle
import time

from satbench.utils.options import add_model_options
from satbench.utils.logger import Logger
from satbench.utils.utils import set_seed, safe_log
from satbench.utils.format_print import FormatTable
from satbench.data.dataloader import get_dataloader
from satbench.models.gnn import GNN
from torch_scatter import scatter_sum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, choices=['satisfiability', 'assignment'], help='Experiment task')
    parser.add_argument('test_dir', type=str, help='Directory with testing data')
    parser.add_argument('checkpoint', type=str, help='Checkpoint to be tested')
    parser.add_argument('--test_splits', type=str, nargs='+', choices=['sat', 'unsat', 'augmented_sat', 'augmented_unsat', 'trimmed'], default=None, help='Validation splits')
    parser.add_argument('--test_sample_size', type=int, default=None, help='The number of instance in validation dataset')
    parser.add_argument('--test_augment_ratio', type=float, default=None, help='The ratio between added clauses and all learned clauses')
    parser.add_argument('--label', type=str, choices=[None, 'satisfiability', 'assignment', 'unsat_core'], default=None, help='Directory with validating data')
    parser.add_argument('--data_fetching', type=str, choices=['parallel', 'sequential'], default='parallel', help='Fetch data in sequential order or in parallel')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for data loading')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')

    add_model_options(parser)
    
    opts = parser.parse_args()

    set_seed(opts.seed)
    
    opts.log_dir = os.path.abspath(os.path.join(opts.checkpoint,  '..', '..'))
    opts.eval_dir = os.path.join(opts.log_dir, 'evaluations')

    difficulty, dataset = tuple(os.path.abspath(opts.test_dir).split(os.path.sep)[-3:-1])
    checkpoint_name = os.path.splitext(os.path.basename(opts.checkpoint))[0]
    os.makedirs(opts.eval_dir, exist_ok=True)

    opts.log = os.path.join(opts.log_dir, f'eval_log_iterations_{opts.n_iterations}.txt')
    sys.stdout = Logger(opts.log, sys.stdout)
    sys.stderr = Logger(opts.log, sys.stderr)

    opts.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(opts)

    model = GNN(opts)
    model.to(opts.device)
    test_loader = get_dataloader(opts.test_dir, opts.test_splits, opts.test_sample_size, opts.test_augment_ratio, opts, 'test')

    print('Loading model checkpoint from %s..' % opts.checkpoint)
    if opts.device.type == 'cpu':
        checkpoint = torch.load(opts.checkpoint, map_location='cpu')
    else:
        checkpoint = torch.load(opts.checkpoint)

    model.load_state_dict(checkpoint['state_dict'])
    model.to(opts.device)

    all_results = []
    test_tot = 0
    test_acc = 0
    rmse = 0
    solved = 0

    if opts.task == 'satisfiability':
        format_table = FormatTable()

    t0 = time.time()

    print('Testing...')
    model.eval()
    for data in test_loader:
        data = data.to(opts.device)
        batch_size = data.num_graphs
        with torch.no_grad():
            if opts.task == 'satisfiability':
                pred = model(data)
                label = data.y
                format_table.update(pred, label)
                all_results.extend(pred.tolist())
            else:
                pass
            
        test_tot += batch_size
    
    if opts.task == 'satisfiability':
        format_table.print_stats()
    elif opts.task == 'assignment':
        pass

    t = time.time() - t0
    print('Solving Time: %f' % t)

    # with open('%s/task=%s_difficulty=%s_dataset=%s_split=%s_checkpoint=%s_n_iterations=%d.pkl' % \
    #     (opts.eval_dir, opts.task, difficulty, dataset, '_'.join(opts.test_splits), checkpoint_name, opts.n_iterations), 'wb') as f:
    #     pickle.dump((all_results, test_acc), f)


if __name__ == '__main__':
    main()
