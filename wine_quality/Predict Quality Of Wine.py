import numpy as np 
import pandas as pd 
from time import time 
from IPython.display import display 
import matplotlib.pyplot as plt 
import seaborn as sns 
#import visuals as vs

data = pd.read_csv("./wine_quality/winequality-red.csv", sep=';')
#print(data.head(5))
#print(data.isnull().any())
#print(data.info())
#print(data.columns)
print(data.shape)

# number of xxx
n_wines = data.shape[0]
quality_above_6 = data.loc[data['quality'] > 6]
n_above_6 = quality_above_6.shape[0]
quality_below_5 = data.loc[data['quality'] < 5]
n_below_5 = quality_below_5.shape[0]
quality_between_5 = data.loc[(data['quality'] >=5) & (data['quality'] <= 6)]
n_between_5 = quality_between_5.shape[0]

greater_precent = n_above_6/n_wines*100 #13.570981863664791

print("Total number of wine data: %d" %(n_wines))
print(f"Wines with rating 7 and above: {n_above_6}")
print("Wines with rating less than 5: {}".format(n_below_5))
print(f"Wines with rating 5 and 6 {n_between_5}")
print(f"Percentage of wines with quality 7 and above:{greater_precent:.2f}")

#display(np.round(data.describe()))
#print(data.describe())

#pd.plotting.scatter_matrix(data, alpha=0.3, figsize = (40,40), diagonal='kde');
#plt.show()
#correlation = data.corr() 
#plt.figure(figsize=(14,12))
#heatmap = sns.heatmap(correlation, annot=True, linewidths=0, vmin=-1, cmap="RdBu_r")
#plt.show()


### check unnormal value ###
Q1 = np.percentile(data['quality'], q=25)
Q3 = np.percentile(data['quality'], q=75)
step = 1.5 * (Q3-Q1)
print(f"Data which going to consider outliers is {data.columns[11]}")
display(data[~((data['quality'] >= Q1 - step) & (data['quality'] <= Q3 + step))])
display(data[(data['quality'] >= Q1 - step) & (data['quality'] <= Q3 + step)])
bad_data =data[~((data['quality'] >= Q1 - step) & (data['quality'] <= Q3 + step))] 
good_data = data.drop(data.index[bad_data.index])
print(good_data)

#data2 = {'Name': ['John', 'Anna', 'Peter', 'Linda'],
#        'Age': [28, 34, 29, 32]}
#df = pd.DataFrame(data2)
#df.index = [8, 9, 10, 11] 
#print(df.iloc[1])

    
