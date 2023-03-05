# ca
# python satbench/generators/ca.py ~/scratch/satbench/easy/ca/train 8000 --min_n 10 --max_n 40
# python satbench/generators/ca.py ~/scratch/satbench/easy/ca/valid 1000 --min_n 10 --max_n 40
# python satbench/generators/ca.py ~/scratch/satbench/easy/ca/test 1000 --min_n 10 --max_n 40

python satbench/generators/ca.py ~/scratch/satbench/medium/ca/train 8000 --min_n 40 --max_n 200
python satbench/generators/ca.py ~/scratch/satbench/medium/ca/valid 1000 --min_n 40 --max_n 200
python satbench/generators/ca.py ~/scratch/satbench/medium/ca/test 1000 --min_n 40 --max_n 200

python satbench/generators/ca.py ~/scratch/satbench/hard/ca/test 1000 --min_n 200 --max_n 400

python satbench/generators/augmented_formula.py ~/scratch/satbench/easy/ca/train --splits sat unsat
python satbench/generators/augmented_formula.py ~/scratch/satbench/easy/ca/valid --splits sat unsat
python satbench/generators/augmented_formula.py ~/scratch/satbench/medium/ca/train --splits sat unsat
python satbench/generators/augmented_formula.py ~/scratch/satbench/medium/ca/valid --splits sat unsat

# ps
python satbench/generators/ps.py ~/scratch/satbench/easy/ps/train 8000 --min_n 10 --max_n 40
python satbench/generators/ps.py ~/scratch/satbench/easy/ps/valid 1000 --min_n 10 --max_n 40
python satbench/generators/ps.py ~/scratch/satbench/easy/ps/test 1000 --min_n 10 --max_n 40

python satbench/generators/ps.py ~/scratch/satbench/medium/ps/train 8000 --min_n 40 --max_n 200
python satbench/generators/ps.py ~/scratch/satbench/medium/ps/valid 1000 --min_n 40 --max_n 200
python satbench/generators/ps.py ~/scratch/satbench/medium/ps/test 1000 --min_n 40 --max_n 200

python satbench/generators/ps.py ~/scratch/satbench/hard/ps/test 1000 --min_n 200 --max_n 400

python satbench/generators/augmented_formula.py ~/scratch/satbench/easy/ps/train --splits sat unsat
python satbench/generators/augmented_formula.py ~/scratch/satbench/easy/ps/valid --splits sat unsat
python satbench/generators/augmented_formula.py ~/scratch/satbench/medium/ps/train --splits sat unsat
python satbench/generators/augmented_formula.py ~/scratch/satbench/medium/ps/valid --splits sat unsat