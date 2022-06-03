#adapted from https://github.com/cfcooney/medium_posts/blob/master/scattered_boxplots.ipynb
#adapted from https://levelup.gitconnected.com/statistics-on-seaborn-plots-with-statannotations-2bfce0394c00
#NEEDS TO BE USED ON THE HPC in envciaran2
#/hpc/local/CentOS7/common/lang/python/3.6.1/bin/python3.6
import sys; sys.path.insert(0, '/home/hers_en/rmclaughlin/tf/lib/python3.6/site-packages') ; sys.path.insert(0, '/hpc/local/CentOS7/modulefiles/python_libs/3.6.1'); sys.path.insert(0, '/hpc/hers_en/rmclaughlin/ciaran/keras_tryout/envciaran2/lib/python3.6/site-packages')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from statannotations.Annotator import Annotator

binary = sys.argv[1]
results_file = sys.argv[2]

str_var = sys.argv[3] #'ft10_1k_300_mlma' #variable to save out boxplot

metric = "$R^{2}$" if binary == 'False' else 'AUC'

data = pd.read_csv(results_file, header=None, index_col=0) #i.e. /external_storage/ciaran/try_folder/sample_results.txt 
#data = data.iloc[:,:4] 
df = data.transpose()
#df = pd.DataFrame(data, columns=['gBLUP', 'Regression','SVM','FNN','Random Forest','LASSO', 'Ridge', 'CNN'])

df = df.reindex(df.mean().sort_values().index, axis=1) # sort by column means
gBLUP  = df.pop('gBLUP') #grab out gBLUP column
df.insert(0, 'gBLUP', gBLUP) #pop back in as first column 
results, names, scatter_points = [],[],[]

for i, col in enumerate(df.columns):
	results.append(df[col].values)
	names.append(col)
	scatter_points.append(np.random.normal(i + 1, 0.04, df[col].values.shape[0])) # adds jitter to the data points - can be adjusted

for i in range(0, len(results)-1):
	results[i][results[i]<0] = 0 #change negative r^2 values to 0 

sns.set_style("whitegrid")
#ax = plt.boxplot(results, labels=names)

#palette = ['r', 'g', 'b', 'y']
#for x, val, colour in zip(scatter_pointa, results, palette):
#	plt.scatter(x, val, alpha=0.4, color=colour)	
pairs =[("Ridge", "gBLUP")]#, ("RBF", "gBLUP"),("LASSO", "gBLUP"),("Ridge", "gBLUP"),("RF", "gBLUP"),("FNN", "gBLUP"),("CNN", "gBLUP")] #Only choose signif pairs!
pvalues = ["p=0.03701","p=0.54854","p=0.00147","p=0.00016","p=0.00014","p=0.00031","p=0.14988"]
sig_levels = ["*"] #"**", "***"

#formatted_pvalues = [f'p={pvalue:.2e}' for pvalue in pvalues]
floop = pd.melt(df)
floop = floop.rename(columns={0: 'Model'})
y = "value"
x = "Model"
ax = sns.boxplot(data=floop, x=x, y=y, palette="Blues") #order = order
ax2 = plt.gca() #get_current_axes
ax2.set_ylim([0, 1])

signif_result = str(input("Any significant results? [ YES or NO ] ")).lower()

if signif_result == 'yes':
        annotator = Annotator(ax, pairs, data=floop, x=x, y=y)
        annotator.set_custom_annotations(sig_levels) # or pavlues
        annotator.annotate()

if signif_result not in ['yes', 'no']:
        print("yes or no required; exiting!")
        exit()


#for x, val in zip(scatter_points, results):
#	plt.scatter(x, val, alpha=0.4, color='b')

plt.title("Nested Cross-Validation Results Between Models", pad=20, fontweight='bold', fontsize='15')
plt.xlabel("Model Type", fontweight='bold', fontsize=11)
plt.ylabel(metric, fontweight='bold', fontsize=16)
plt.show()
plt.savefig('boxplot_model_results_' + str_var, dpi=300)
plt.clf(); plt.close()

'''
df1 = df1.mean() #returns mean of each model (8,1)shape
df2 = df2.mean()
df.mean().mean() # returns mean of entire dataframe
df = pd.concat([df1,df2], axis=1) #merge results across snp sets
df.columns = ['10,000', '5000']
#df = pd.DataFrame(data, columns=['100', '1000', '5000', '10,000'])
results, names, scatter_points = [],[],[]
for i, col in enumerate(df.columns):
        results.append(df[col].values)
        names.append(col)
        scatter_points.append(np.random.normal(i + 1, 0.04, df[col].values.shape[0])) # adds jitter to the data points - can be adjusted


sns.set_style("whitegrid")
ax = plt.boxplot(results, labels=names)
for x, val in zip(scatter_points, results):
        plt.scatter(x, val, alpha=0.4, color='b')

plt.title("Number of Features vs Model Performance", fontweight='bold', fontsize='15')
plt.xlabel("SNP Set Size", fontweight='normal', fontsize=11)
plt.ylabel("Average %s" % metric, fontweight='normal', fontsize=11)
plt.show()
plt.savefig("boxplot_snpset_results_nonpara", dpi=300)
plt.clf(); plt.close()
'''


