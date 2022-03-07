#/hpc/local/CentOS7/common/lang/python/3.6.1/bin/python3.6
#ARG1 = Phenotype and Covariate file with Header
print("Please Print to a log...")
import sys; sys.path.insert(0, '/home/hers_en/rmclaughlin/tf/lib/python3.6/site-packages') ; sys.path.insert(0, '/hpc/local/CentOS7/modulefiles/python_libs/3.6.1'); sys.path.insert(0, '/hpc/hers_en/rmclaughlin/ciaran/keras_tryout/envciaran2/lib/python3.6/site-packages')
import numpy as np
import statsmodels.api as sm
data = np.loadtxt(sys.argv[1], skiprows=1)
if np.array_equal(data[:,1], data[:,1].astype(bool)) == True:
	print("QUANTATATIVE PHENOTYPE - LINEAR REGRESSION")
	model = sm.regression.linear_model.OLS(data[:, 3],sm.add_constant(data[:,1:2]))
else:
	print("BINARY PHENOTYPE - LOGISTIC REGRESSION")
	model = sm.logit(data[:, 3],sm.add_constant(data[:,1:2]))

result = model.fit()
print(result.summary())
y_resids = result.resid_pearson
print("Generating Pearson Residuals...")
np.savetxt((str(sys.argv[1]) + ".resids"), y_resids)


