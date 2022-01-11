# machine_learning2
Sep 2021
please note pkl files have X and y attributes that correspons to the raw file (e.g trial.raw) fed in the original bash command. Not really used in the actual fitting. 


import subprocess, os


print(subprocess.check_output(["git", "describe", "--always"], cwd=os.path.dirname(os.path.abspath(__file__))).strip().decode())

##EXAMPLE ##python cv_grid_all_ml.py FT16_cv_br  nahnah trial.raw  shuf FT16cv5k 506

import ast
goal_dict = {'degree':{},'epsilon':{},'C':{},'kernel':{},'gamma':{}}
>>> with open('cv_grid_all_ml_70_shuf_10k_ft10.txt.scoredict','r') as f:
...     for line in f:
...             dfg = ast.literal_eval(str(line.split("\t")[0]))
...             for i in dfg:
...                     if dfg[i] in goal_dict[i]:
...                             goal_dict[i][dfg[i]].append(float(line.split("\t")[1].strip()))
...                     else:
...                             goal_dict[i][dfg[i]] = []
...                             goal_dict[i][dfg[i]].append(float(line.split("\t")[1].strip()))

	

grep -B4 'Typer LinearSVR' cv_grid_all_ml_70_shuf_10k_ft10.txt | grep -v '{' | grep '0' > cv_grid_all_ml_70_shuf_10k_ft10.txt.score
pr -tmJ cv_grid_all_ml_70_shuf_10k_ft10.txt.svrdict cv_grid_all_ml_70_shuf_10k_ft10.txt.score > c.txt
sed -e "s/\r//g" c.txt | head

 
