import argparse
import glob
import os

from satbench.utils.utils import parse_cnf_file, hash_clauses
from tqdm import tqdm


def check_clauses(clauses):
    c = [frozenset(clause) for clause in clauses]
    return len(c) == len(frozenset(c))


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
    cnt = 0

    for f in tqdm(all_files):
        n_vars, clauses = parse_cnf_file(f)
        if not check_clauses(clauses):
            cs = []
            for c in clauses:
                if frozenset(c) not in cs:
                    cs.append(frozenset(c))
                else:
                    print(f)
                    print(c)
                    exit()
            cnt += 1

    print(cnt)


if __name__ == '__main__':
    main()
