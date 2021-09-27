import numpy as np
import sklearn.model_selection
import sys
import subprocess, os
print(subprocess.check_output(["git", "describe", "--always"], cwd=os.path.dirname(os.path.abspath(__file__))).strip().decode())
for i in range(1,len(sys.argv)):
	print(sys.argv[i])


my_cv = sklearn.model_selection.KFold(n_splits=int(sys.argv[1]), shuffle=True, random_state=42)
data = np.loadtxt(sys.argv[2], usecols=[0,1], dtype='str')
print("Assuming there is no header...")

cv_obj = my_cv.split(X=data)

count = 0 #establish count for saving out cv

for train_index, test_index in my_cv.split(X=data):
	with open('train_split_cv_' + str(count+1) + '.txt', 'w') as f:
		for item in train_index:
			f.write("%s\n" % item)
	with open('test_split_cv_' + str(count+1) + '.txt', 'w') as f2:
		for item in test_index:
			f2.write("%s\n" % item)
	count += 1
  
  
  
      
      
