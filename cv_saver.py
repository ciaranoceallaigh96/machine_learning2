import numpy as np
import sklearn.model_selection

my_cv = sklearn.model_selection.KFold(n_splits=sys.argv[1], shuffle=True, random_state=42)
data = np.loadtxt(sys.argv[2], usecols=[0,1], dtype='str')
cv_obj = my_cv.split(X=data)

count = 0 #establish count for saving out cv

for train_index, test_index in my_cv.split(X=data):
  with open(('train_split_cv_' + str(count) '.txt'), 'w') as f:
    for item in train_index:
      f.write("%s\n" % item)
  with open(('test_split_cv_' + str(count) '.txt'), 'w') as f2:
    for item in test_index:
      f2.write("%s\n" % item)
  count +=1
  
  
  
      
      
