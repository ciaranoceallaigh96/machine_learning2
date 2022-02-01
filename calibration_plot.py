#Generates Calibration Plot for class prediction models (model needs to be able to output probabilities)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.calibration import calibration_curve
import sys

model = sys.argv[1]
validation_predictions = model.predict_proba(x_val)

y, x = calibration_curve(label_test, validation_predictions[:,1], n_bins=10) #standard number of bins=10 #approach and code adpaetd from Chang Hsin Lee (changhsinlee.com/python-calibration-plot/)

fig, ax = plt.subplots()
plt.plot(x,y, marker='o', linewidth=1, label='model_type')
# reference line, legends, and axis labels
line = mlines.Line2D([0, 1], [0, 1], color='black')
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
fig.suptitle('Calibration Plot for ?')
ax.set_xlabel('Predicted probability')
ax.set_ylabel('True probability (per bin)')
plt.clf()
plt.close()

'''
#plot training curve
plt.plot(object.history['acc'])
plt.plot(object.history['val_acc'])
plt.title('Model Accuracy'); plt.ylabel('Accuracy'); plt.xlabel('Epoch'); plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
plt.savefig('acc_training_curve_conf_adj_regress_on_pheno_mixed_chr1_grid3' + str(date_object), dpi=300)
plt.clf()
plt.close()
'''
