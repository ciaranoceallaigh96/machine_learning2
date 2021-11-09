# machine_learning2
Sep 2021
please note pkl files have X and y attributes that correspons to the raw file (e.g trial.raw) fed in the original bash command. Not really used in the actual fitting. 


import subprocess, os


print(subprocess.check_output(["git", "describe", "--always"], cwd=os.path.dirname(os.path.abspath(__file__))).strip().decode())

##EXAMPLE ##python cv_grid_all_ml.py FT16_cv_br  nahnah trial.raw  shuf FT16cv5k 506 
