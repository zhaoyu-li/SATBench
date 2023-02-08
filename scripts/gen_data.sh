# easy
python satbench/generators/sr.py ~/scratch/satbench/easy/sr/train 8000 --min_n 10 --max_n 200
python satbench/generators/sr.py ~/scratch/satbench/easy/sr/valid 1000 --min_n 10 --max_n 200
python satbench/generators/sr.py ~/scratch/satbench/easy/sr/test 1000 --min_n 10 --max_n 200

python satbench/generators/3-sat.py ~/scratch/satbench/easy/3-sat/train 8000 --min_n 10 --max_n 100
python satbench/generators/3-sat.py ~/scratch/satbench/easy/3-sat/valid 1000 --min_n 10 --max_n 100
python satbench/generators/3-sat.py ~/scratch/satbench/easy/3-sat/test 1000 --min_n 10 --max_n 100

# medium
python satbench/generators/sr.py ~/scratch/satbench/medium/sr/train 8000 --min_n 200 --max_n 400
python satbench/generators/sr.py ~/scratch/satbench/medium/sr/valid 1000 --min_n 200 --max_n 400
python satbench/generators/sr.py ~/scratch/satbench/medium/sr/test 1000 --min_n 200 --max_n 400

python satbench/generators/3-sat.py ~/scratch/satbench/medium/3-sat/train 8000 --min_n 100 --max_n 200
python satbench/generators/3-sat.py ~/scratch/satbench/medium/3-sat/valid 1000 --min_n 100 --max_n 200
python satbench/generators/3-sat.py ~/scratch/satbench/medium/3-sat/test 1000 --min_n 100 --max_n 200

# hard
python satbench/generators/sr.py ~/scratch/satbench/hard/sr/test 1000 --min_n 400 --max_n 800
python satbench/generators/3-sat.py ~/scratch/satbench/hard/3-sat/test 1000 --min_n 200 --max_n 400