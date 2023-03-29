import os
import argparse
import sympy
import networkx as nx


from concurrent.futures.process import ProcessPoolExecutor
from pysat.solvers import Cadical
from satbench.utils.utils import ROOT_DIR, parse_cnf_file, VIG
from satbench.external.satfactor import generate_instance_known_factors


class Generator:
    def __init__(self, opts):
        self.opts = opts
        self.opts.sat_out_dir = os.path.join(self.opts.out_dir, 'sat')
        os.makedirs(self.opts.sat_out_dir, exist_ok=True)
        
    def run(self, t):
        if t % self.opts.print_interval == 0:
            print('Generating instance %d.' % t)
        
        factor1 = sympy.randprime(pow(2, self.opts.min_b), pow(2, self.opts.max_b))
        factor2 = sympy.randprime(pow(2, self.opts.min_b), pow(2, self.opts.max_b))

        cnf_filepath = os.path.abspath(os.path.join(self.opts.sat_out_dir, '%.5d.cnf' % (t)))
        dimacs = generate_instance_known_factors(factor1, factor2)
        
        with open(cnf_filepath, 'w') as f:
            f.write(dimacs)
        
        n_vars, clauses = parse_cnf_file(cnf_filepath)
        solver = Cadical(bootstrap_with=clauses)
        sat = solver.solve()
        assert sat == True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out_dir', type=str)
    parser.add_argument('n_instances', type=int)

    parser.add_argument('--min_b', type=int, default=5)
    parser.add_argument('--max_b', type=int, default=20)

    parser.add_argument('--print_interval', type=int, default=1000)

    parser.add_argument('--n_process', type=int, default=10, help='Number of processes to run')

    opts = parser.parse_args()

    generator = Generator(opts)
    
    with ProcessPoolExecutor(max_workers=opts.n_process) as pool:
        pool.map(generator.run, range(opts.n_instances))


if __name__ == '__main__':
    main()
