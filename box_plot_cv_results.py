#adapted from https://github.com/cfcooney/medium_posts/blob/master/scattered_boxplots.ipynb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

str_var = ' ' #variable to save out boxplot

metric = 'R^2' if binary == 'False' else 'AUC'

data = pd.read_csv('sample_results.txt', header=None, index_col=0)
df = data.transpose()
#df = pd.DataFrame(data, columns=['gBLUP', 'Regression','SVM','FNN','Random Forest','LASSO', 'Ridge', 'CNN'])


results, names, scatter_points = [],[],[]

for i, col in enumerate(df.columns):
	results.append(df[col].values)
	names.append(col)
	scatter_points.append(np.random.normal(i + 1, 0.04, df[col].values.shape[0])) # adds jitter to the data points - can be adjusted

sns.set_style("whitegrid")
plt.boxplot(results, labels=names)
#palette = ['r', 'g', 'b', 'y']
#for x, val, colour in zip(scatter_pointa, results, palette):
#	plt.scatter(x, val, alpha=0.4, color=colour)	
for x, val in zip(scatter_points, results):
	plt.scatter(x, val, alpha=0.4, color='b')

plt.title("Nested Cross-Validation Results Between Models", fontweight='bold', fontsize='15')
plt.xlabel("Model Type", fontweight='normal', fontsize=11)
plt.ylabel(metric, fontweight='normal', fontsize=11)
plt.show()
plt.savefig('boxplot_model_results_' + str_var, dpi=300)
plt.clf(); plt.close()

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
plt.boxplot(results, labels=names)
for x, val in zip(scatter_points, results):
        plt.scatter(x, val, alpha=0.4, color='b')

plt.title("Number of Features vs Model Performance", fontweight='bold', fontsize='15')
plt.xlabel("SNP Set Size", fontweight='normal', fontsize=11)
plt.ylabel("Average %s" % metric, fontweight='normal', fontsize=11)
plt.show()
plt.savefig("boxplot_snpset_results", dpi=300)
plt.clf(); plt.close()

