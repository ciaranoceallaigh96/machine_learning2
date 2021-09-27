# machine_learning2
Sep 2021

import subprocess, os
print(subprocess.check_output(["git", "describe", "--always"], cwd=os.path.dirname(os.path.abspath(__file__))).strip().decode())
