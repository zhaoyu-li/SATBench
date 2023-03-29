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
        
        while True:
            # Number of factorys
            f = random.randint(self.opts.min_f, self.opts.max_f)
            # Number of workers
            w = random.randint(self.opts.min_w, self.opts.max_w)
            # Number of jobs
            j = random.randint(self.opts.min_j, self.opts.max_j)
            # The probability of a worker can do a job
            p = random.uniform(self.opts.min_p, self.opts.max_p)
            
            table_filepath = os.path.abspath(os.path.join(self.opts.out_dir, '%.5d.txt' % (t)))
            cmd_line = ['./wdp2table', str(f), str(w), str(j), str(p)]
            with open(table_filepath, 'w') as f_out:
                try:
                    process = subprocess.Popen(cmd_line, stdout=f_out, stderr=f_out, cwd=self.exec_dir, start_new_session=True)
                    process.communicate()
                except:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            
            if os.stat(table_filepath).st_size == 0:
                os.remove(table_filepath)
                continue

            cnf_filepath = os.path.abspath(os.path.join(self.opts.out_dir, '%.5d.cnf' % (t)))
            cmd_line = ['./table2cnf', str(f), str(w), str(j), table_filepath]

            with open(cnf_filepath, 'w') as f_out:
                try:
                    process = subprocess.Popen(cmd_line, stdout=f_out, stderr=f_out, cwd=self.exec_dir, start_new_session=True)
                    process.communicate()
                except:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            
            if os.stat(cnf_filepath).st_size == 0:
                os.remove(cnf_filepath)
                continue
            
            n_vars, clauses = parse_cnf_file(cnf_filepath)
            vig = VIG(n_vars, clauses)
            if not nx.is_connected(vig):
                os.remove(table_filepath)
                os.remove(cnf_filepath)
                continue

            solver = Cadical(bootstrap_with=clauses)

            if solver.solve():
                write_dimacs_to(n_vars, clauses, os.path.join(self.opts.sat_out_dir, '%.5d.cnf' % (t)))
                os.remove(table_filepath)
                os.remove(cnf_filepath)
                break
            

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out_dir', type=str)
    parser.add_argument('n_instances', type=int)

    parser.add_argument('--min_f', type=int, default=15)
    parser.add_argument('--max_f', type=int, default=25)

    parser.add_argument('--min_w', type=int, default=35)
    parser.add_argument('--max_w', type=int, default=45)

    parser.add_argument('--min_j', type=int, default=15)
    parser.add_argument('--max_j', type=int, default=25)

    parser.add_argument('--min_p', type=float, default=0.85)
    parser.add_argument('--max_p', type=float, default=0.95)

    parser.add_argument('--print_interval', type=int, default=1000)

    parser.add_argument('--n_process', type=int, default=10, help='Number of processes to run')

    opts = parser.parse_args()

    generator = Generator(opts)
    
    with ProcessPoolExecutor(max_workers=opts.n_process) as pool:
        pool.map(generator.run, range(opts.n_instances))
    
    # for i in range(opts.n_instances):
    #     generator.run(i)


if __name__ == '__main__':
    main()
