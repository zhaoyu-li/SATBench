import argparse
import glob
import os
import networkx as nx
import networkx.algorithms.community as nx_comm
import glob

from tqdm import tqdm
from satbench.utils.utils import parse_cnf_file, VIG, VCG, clean_clauses
from collections import defaultdict


terms = ['n_vars', 'n_clauses', 'vig-clustering_coefficient', 'vig-modularity', 'vcg-clustering_coefficient', 'vcg-modularity']

# terms = ['n_vars', 'n_clauses']

def calc_stats(f):
    n_vars, clauses = parse_cnf_file(f)
    clauses = clean_clauses(clauses)
    vig = VIG(n_vars, clauses)
    vcg = VCG(n_vars, clauses)

    is_connected_vig =  nx.is_connected(vig)
    is_connected_vcg =  nx.is_connected(vcg)

    return {
        'n_vars': n_vars,
        'n_clauses': len(clauses),
        'vig-clustering_coefficient': nx.average_clustering(vig) if is_connected_vig else -1,
        'vig-modularity': nx_comm.modularity(vig, nx_comm.louvain_communities(vig)) if is_connected_vig else -1,
        'vcg-clustering_coefficient': nx.average_clustering(vcg) if is_connected_vcg else -1,
        'vcg-modularity': nx_comm.modularity(vcg, nx_comm.louvain_communities(vcg)) if is_connected_vcg else -1
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='Directory with sat data')
    opts = parser.parse_args()
    print(opts)
    
    print('Calculating statistics...')
    
    all_files = sorted(glob.glob(opts.data_dir + '/**/*.cnf', recursive=True))
    all_files = [os.path.abspath(f) for f in all_files if 'augmented' not in f]

    stats = defaultdict(int)
    
    for f in tqdm(all_files):
        s = calc_stats(f)
        for t in terms:
            stats[t] += s[t]
    
    for t in terms:
        print('%30s\t%10.2f' % (t, stats[t] / len(all_files)))
    

if __name__ == '__main__':
    main()
