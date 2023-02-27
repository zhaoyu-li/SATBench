# sr
python satbench/generators/sr.py ~/scratch/satbench/easy/sr/train 8000 --min_n 10 --max_n 40
python satbench/generators/sr.py ~/scratch/satbench/easy/sr/valid 1000 --min_n 10 --max_n 40
python satbench/generators/sr.py ~/scratch/satbench/easy/sr/test 1000 --min_n 10 --max_n 40

python satbench/generators/sr.py ~/scratch/satbench/medium/sr/train 8000 --min_n 40 --max_n 200
python satbench/generators/sr.py ~/scratch/satbench/medium/sr/valid 1000 --min_n 40 --max_n 200
python satbench/generators/sr.py ~/scratch/satbench/medium/sr/test 1000 --min_n 40 --max_n 200

# python satbench/generators/sr.py ~/scratch/satbench/hard/sr/test 1000 --min_n 200 --max_n 400

python satbench/generators/augmented_formula.py ~/scratch/satbench/easy/sr/train --splits sat unsat
python satbench/generators/augmented_formula.py ~/scratch/satbench/easy/sr/valid --splits sat unsat
python satbench/generators/augmented_formula.py ~/scratch/satbench/medium/sr/train --splits sat unsat
python satbench/generators/augmented_formula.py ~/scratch/satbench/medium/sr/valid --splits sat unsat

# 3-sat
python satbench/generators/3-sat.py ~/scratch/satbench/easy/3-sat/train 8000 --min_n 10 --max_n 40
python satbench/generators/3-sat.py ~/scratch/satbench/easy/3-sat/valid 1000 --min_n 10 --max_n 40
python satbench/generators/3-sat.py ~/scratch/satbench/easy/3-sat/test 1000 --min_n 10 --max_n 40

python satbench/generators/3-sat.py ~/scratch/satbench/medium/3-sat/train 8000 --min_n 40 --max_n 200
python satbench/generators/3-sat.py ~/scratch/satbench/medium/3-sat/valid 1000 --min_n 40 --max_n 200
python satbench/generators/3-sat.py ~/scratch/satbench/medium/3-sat/test 1000 --min_n 40 --max_n 200

# python satbench/generators/3-sat.py ~/scratch/satbench/hard/3-sat/test 1000 --min_n 200 --max_n 400

python satbench/generators/augmented_formula.py ~/scratch/satbench/easy/3-sat/train --splits sat unsat
python satbench/generators/augmented_formula.py ~/scratch/satbench/easy/3-sat/valid --splits sat unsat
python satbench/generators/augmented_formula.py ~/scratch/satbench/medium/3-sat/train --splits sat unsat
python satbench/generators/augmented_formula.py ~/scratch/satbench/medium/3-sat/valid --splits sat unsat