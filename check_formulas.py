import argparse
import glob
import os

from satbench.utils.utils import parse_cnf_file, hash_clauses
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='Directory with sat data')
    opts = parser.parse_args()
    print(opts)

    print('Checking duplicated formulas...')

    all_files = sorted(glob.glob(opts.data_dir + '/**/*.cnf', recursive=True))
    all_files = [os.path.abspath(f) for f in all_files]

    print(f'There are {len(all_files)} files.')
    
    hash_list = []

    for f in tqdm(all_files):
        n_vars, clauses = parse_cnf_file(f)
        h = hash_clauses(clauses)
        
        if h not in hash_list:
            hash_list.append(h)
        else:
            print('There are two same CNF formulas!')
            print(f)
            print(all_files[hash_list.index(h)])
            break


if __name__ == '__main__':
    main()
