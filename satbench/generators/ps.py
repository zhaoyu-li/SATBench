import os
import argparse
import random
import subprocess
import networkx as nx

from concurrent.futures.process import ProcessPoolExecutor
from pysat.solvers import Cadical
from satbench.utils.utils import ROOT_DIR, parse_cnf_file, write_dimacs_to, VIG


class Generator:
    def __init__(self, opts):
        self.opts = opts
        self.opts.sat_out_dir = os.path.join(self.opts.out_dir, 'sat')
        self.opts.unsat_out_dir = os.path.join(self.opts.out_dir, 'unsat')
        self.exec_dir = os.path.join(ROOT_DIR, 'external')
        os.makedirs(self.opts.sat_out_dir, exist_ok=True)
        os.makedirs(self.opts.unsat_out_dir, exist_ok=True)
        
    def run(self, t):
        if t % self.opts.print_interval == 0:
            print('Generating instance %d.' % t)

        sat = False
        unsat = False
        
        while not sat or not unsat:
            k = random.randint(self.opts.min_k, self.opts.max_k)
            n_vars = random.randint(self.opts.min_n, self.opts.max_n)
            r = random.uniform(6, 8)
            n_clauses = int(r * n_vars)
            b = random.uniform(self.opts.min_b, self.opts.max_b)
            T = random.uniform(self.opts.min_T, self.opts.max_T)
            
            cnf_filepath = os.path.abspath(os.path.join(self.opts.out_dir, '%.5d.cnf' % (t)))
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

            solver = Cadical(bootstrap_with=clauses)
            
            if solver.solve():
                if not sat:
                    sat = True
                    write_dimacs_to(n_vars, clauses, os.path.join(self.opts.sat_out_dir, '%.5d.cnf' % (t)))
            else:
                if not unsat:
                    unsat = True
                    write_dimacs_to(n_vars, clauses, os.path.join(self.opts.unsat_out_dir, '%.5d.cnf' % (t)))

            os.remove(cnf_filepath)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out_dir', type=str)
    parser.add_argument('n_instances', type=int)

    parser.add_argument('--min_k', type=int, default=4)
    parser.add_argument('--max_k', type=int, default=5)

    parser.add_argument('--min_n', type=int, default=10)
    parser.add_argument('--max_n', type=int, default=40)

    parser.add_argument('--min_b', type=float, default=0)
    parser.add_argument('--max_b', type=float, default=1)

    parser.add_argument('--min_T', type=float, default=0.75)
    parser.add_argument('--max_T', type=float, default=1.5)

    parser.add_argument('--print_interval', type=int, default=1000)

    parser.add_argument('--n_process', type=int, default=10, help='Number of processes to run')

    opts = parser.parse_args()

    generator = Generator(opts)
    
    # with ProcessPoolExecutor(max_workers=opts.n_process) as pool:
    #     pool.map(generator.run, range(opts.n_instances))
    
    for i in range(opts.n_instances):
        generator.run(i)
    

if __name__ == '__main__':
    main()
