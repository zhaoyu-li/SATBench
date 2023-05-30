import os
import math
import argparse
import random
import networkx as nx

from pysat.solvers import Cadical
from cnfgen import VertexCoverFormula
from satbench.utils.utils import write_dimacs_to, VIG, clean_clauses, hash_clauses
from tqdm import tqdm
from scipy.optimize import fsolve


sat_cnt = 0
unsat_cnt = 0


class Generator:
    def __init__(self, opts):
        self.opts = opts
        self.hash_list = []

    def run(self):
        for split in ['train', 'valid', 'test']:
            n_instances = getattr(self.opts, f'{split}_instances')
            if n_instances > 0:
                sat_out_dir = os.path.join(os.path.abspath(self.opts.out_dir), f'{split}/sat')
                unsat_out_dir = os.path.join(os.path.abspath(self.opts.out_dir), f'{split}/unsat')
                os.makedirs(sat_out_dir, exist_ok=True)
                os.makedirs(unsat_out_dir, exist_ok=True)
                print(f'Generating k-vercov {split} set...')
                for i in tqdm(range(n_instances)):
                    self.generate(i, sat_out_dir, unsat_out_dir)
    
    def generate(self, i, sat_out_dir, unsat_out_dir):
        sat = False
        unsat = False

        # ensure k is uniformly sampled
        k = random.randint(self.opts.min_k, self.opts.max_k)
        
        while not sat or not unsat:
            v = random.randint(self.opts.min_v, self.opts.max_v)
            if v - k <= 2:
                v = v + 1
            com_k = v - k
            p = pow(1/math.comb(v,com_k), 2/(com_k*(com_k-1)))
            com_graph = nx.generators.erdos_renyi_graph(v, p=p)
            graph = nx.complement(com_graph)

            if not nx.is_connected(graph):
                continue
            
            cnf = VertexCoverFormula(graph, k)
            n_vars = len(list(cnf.variables()))
            clauses = list(cnf.clauses())
            clauses = [list(cnf._compress_clause(clause)) for clause in clauses]
            vig = VIG(n_vars, clauses)
            if not nx.is_connected(vig):
                continue

            clauses = clean_clauses(clauses)
            h = hash_clauses(clauses)

            if h in self.hash_list:
                continue

            solver = Cadical(bootstrap_with=clauses)
            
            if solver.solve():
                global sat_cnt 
                sat_cnt += 1
                if not sat:
                    sat = True
                    self.hash_list.append(h)
                    write_dimacs_to(n_vars, clauses, os.path.join(sat_out_dir, '%.5d.cnf' % (i)))
            else:
                global unsat_cnt
                unsat_cnt += 1
                if not unsat:
                    unsat = True
                    self.hash_list.append(h)
                    write_dimacs_to(n_vars, clauses, os.path.join(unsat_out_dir, '%.5d.cnf' % (i)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out_dir', type=str)
    
    parser.add_argument('--train_instances', type=int, default=0)
    parser.add_argument('--valid_instances', type=int, default=0)
    parser.add_argument('--test_instances', type=int, default=0)

    parser.add_argument('--min_k', type=int, default=3)
    parser.add_argument('--max_k', type=int, default=5)

    parser.add_argument('--min_v', type=int, default=5)
    parser.add_argument('--max_v', type=int, default=20)
    
    parser.add_argument('--seed', type=int, default=0)

    opts = parser.parse_args()

    random.seed(opts.seed)

    generator = Generator(opts)
    generator.run()

    print(sat_cnt)
    print(unsat_cnt)


if __name__ == '__main__':
    main()
