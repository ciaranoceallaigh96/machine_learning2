import subprocess, os
import sys
import numpy as np
print(subprocess.check_output(["git", "describe", "--always"], cwd=os.path.dirname(os.path.abspath(file))).strip().decode())
print(sys.argv[1])
data = np.loadtxt(sys.argv[1], skiprows=1, usecols=[2,5])
r2_score = str(np.corrcoef(data[:,0], data[:,1]))
print("Result for cv %s is R2 %s" % (sys.argv[2], r2_score))
