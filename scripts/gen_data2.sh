# k-clique
python satbench/generators/k-clique.py ~/scratch/satbench/easy/k-clique/train 80000 --min_v 5 --max_v 20
python satbench/generators/k-clique.py ~/scratch/satbench/easy/k-clique/valid 10000 --min_v 5 --max_v 20
python satbench/generators/k-clique.py ~/scratch/satbench/easy/k-clique/test 10000 --min_v 5 --max_v 20

python satbench/generators/k-clique.py ~/scratch/satbench/medium/k-clique/train 80000 --min_v 20 --max_v 50
python satbench/generators/k-clique.py ~/scratch/satbench/medium/k-clique/valid 10000 --min_v 20 --max_v 50
python satbench/generators/k-clique.py ~/scratch/satbench/medium/k-clique/test 10000 --min_v 20 --max_v 50

python satbench/generators/k-clique.py ~/scratch/satbench/hard/k-clique/test 10000 --min_v 50 --max_v 100

# k-color
python satbench/generators/k-color.py ~/scratch/satbench/easy/k-color/train 80000 --min_v 5 --max_v 20
python satbench/generators/k-color.py ~/scratch/satbench/easy/k-color/valid 10000 --min_v 5 --max_v 20
python satbench/generators/k-color.py ~/scratch/satbench/easy/k-color/test 10000 --min_v 5 --max_v 20

python satbench/generators/k-color.py ~/scratch/satbench/medium/k-color/train 80000 --min_v 20 --max_v 50
python satbench/generators/k-color.py ~/scratch/satbench/medium/k-color/valid 10000 --min_v 20 --max_v 50
python satbench/generators/k-color.py ~/scratch/satbench/medium/k-color/test 10000 --min_v 20 --max_v 50

python satbench/generators/k-color.py ~/scratch/satbench/hard/k-color/test 10000 --min_v 50 --max_v 100

# k-domset
python satbench/generators/k-domset.py ~/scratch/satbench/easy/k-domset/train 80000 --min_v 5 --max_v 20
python satbench/generators/k-domset.py ~/scratch/satbench/easy/k-domset/valid 10000 --min_v 5 --max_v 20
python satbench/generators/k-domset.py ~/scratch/satbench/easy/k-domset/test 10000 --min_v 5 --max_v 20

python satbench/generators/k-domset.py ~/scratch/satbench/medium/k-domset/train 80000 --min_v 20 --max_v 50
python satbench/generators/k-domset.py ~/scratch/satbench/medium/k-domset/valid 10000 --min_v 20 --max_v 50
python satbench/generators/k-domset.py ~/scratch/satbench/medium/k-domset/test 10000 --min_v 20 --max_v 50

python satbench/generators/k-domset.py ~/scratch/satbench/hard/k-domset/test 10000 --min_v 50 --max_v 100