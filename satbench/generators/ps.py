import os
import argparse
import random
import subprocess
import networkx as nx

from pysat.solvers import Cadical
from satbench.utils.utils import ROOT_DIR, parse_cnf_file, write_dimacs_to, VIG, clean_clauses, hash_clauses
from tqdm import tqdm


class Generator:
    def __init__(self, opts):
        self.opts = opts
        self.hash_list = []
        self.exec_dir = os.path.join(ROOT_DIR, 'external')

    def run(self):
        for split in ['train', 'valid', 'test']:
            n_instances = getattr(self.opts, f'{split}_instances')
            if n_instances > 0:
                sat_out_dir = os.path.join(os.path.abspath(self.opts.out_dir), f'{split}/sat')
                unsat_out_dir = os.path.join(os.path.abspath(self.opts.out_dir), f'{split}/unsat')
                os.makedirs(sat_out_dir, exist_ok=True)
                os.makedirs(unsat_out_dir, exist_ok=True)
                print(f'Generating ps {split} set...')
                for i in tqdm(range(n_instances)):
                    self.generate(i, sat_out_dir, unsat_out_dir)
    
    def generate(self, i, sat_out_dir, unsat_out_dir):
        sat = False
        unsat = False
        
        while not sat or not unsat:
            k = random.randint(self.opts.min_k, self.opts.max_k)
            n_vars = random.randint(self.opts.min_n, self.opts.max_n)
            r = random.uniform(6, 8)
            n_clauses = int(r * n_vars)
            b = random.uniform(self.opts.min_b, self.opts.max_b)
            T = random.uniform(self.opts.min_T, self.opts.max_T)
            
            cnf_filepath = os.path.abspath(os.path.join(self.opts.out_dir, '%.5d.cnf' % (i)))
            cmd_line = ['./ps', '-n', str(n_vars), '-m', str(n_clauses), '-k', str(k-3), \
                '-b', str(b), '-T', str(T), '-s', str(random.randint(0, 2**32))]
            
            with open(cnf_filepath, 'w') as f:
                try:
                    process = subprocess.Popen(cmd_line, stdout=f, stderr=f, cwd=self.exec_dir, start_new_session=True)
                    process.communicate()
                except:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            
            if os.stat(cnf_filepath).st_size == 0:
                os.remove(cnf_filepath)
                continue
            
            n_vars, clauses = parse_cnf_file(cnf_filepath)
            vig = VIG(n_vars, clauses)
            if not nx.is_connected(vig):
                os.remove(cnf_filepath)
                continue

            clauses = clean_clauses(clauses)
            h = hash_clauses(clauses)

            if h in self.hash_list:
                continue

            solver = Cadical(bootstrap_with=clauses)
            
            if solver.solve():
                if not sat:
                    sat = True
                    self.hash_list.append(h)
                    write_dimacs_to(n_vars, clauses, os.path.join(sat_out_dir, '%.5d.cnf' % (i)))
            else:
                if not unsat:
                    unsat = True
                    self.hash_list.append(h)
                    write_dimacs_to(n_vars, clauses, os.path.join(unsat_out_dir, '%.5d.cnf' % (i)))

            os.remove(cnf_filepath)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out_dir', type=str)

    parser.add_argument('--train_instances', type=int, default=0)
    parser.add_argument('--valid_instances', type=int, default=0)
    parser.add_argument('--test_instances', type=int, default=0)

    parser.add_argument('--min_k', type=int, default=4)
    parser.add_argument('--max_k', type=int, default=5)

    parser.add_argument('--min_n', type=int, default=10)
    parser.add_argument('--max_n', type=int, default=40)

    parser.add_argument('--min_b', type=float, default=0)
    parser.add_argument('--max_b', type=float, default=1)

    parser.add_argument('--min_T', type=float, default=0.75)
    parser.add_argument('--max_T', type=float, default=1.5)

    parser.add_argument('--seed', type=int, default=0)

    opts = parser.parse_args()

    random.seed(opts.seed)

    generator = Generator(opts)
    generator.run()


if __name__ == '__main__':
    main()
