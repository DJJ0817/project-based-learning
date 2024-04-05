import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib import cm
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import seaborn as sns
import pylab as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

fruits = pd.read_table('fruit_data_with_colors.txt')
print(fruits.head())
print(fruits.shape)
#print(fruits.info())
print(fruits['fruit_name'].unique()) #I can speak mandarin lol
print(fruits.groupby('fruit_name').size())

#sns.countplot(x=fruits['fruit_name'], palette="bright")
#plt.show()

### visualization ###
##print(fruits.drop(2, axis=0))
#fruits.drop('fruit_label', axis=1).plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False, figsize=(9,9))
#plt.show()

print(fruits.drop(['fruit_label','fruit_subtype', 'fruit_name'], axis=1).columns)

X = fruits[fruits.drop(['fruit_label','fruit_subtype', 'fruit_name'], axis=1).columns]
y = fruits['fruit_label']

#map = cm.get_cmap('gnuplot') # set a colormap to 'cmap' 
#scatter = pd.plotting.scatter_matrix(X, c=y, marker='o', s=40, hist_kwds={'bins':15}, figsize=(9,9), cmap=cmap)
#plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

scaler = MinMaxScaler() #scale data to 0~1
X_train = scaler.fit_transform(X_train) # use fit_transform is effective, and portable
X_test = scaler.transform(X_test) #scaler.transform uses the fit_transform pattern, means it might larger than 1

### logistic regression ###
logreg = LogisticRegression()
logreg.fit(X_train, y_train) 
print(f'Accuracy of reg training set {logreg.score(X_train, y_train)}')
print(f'Accuracy of reg test set {logreg.score(X_test, y_test)}')

### decision tree ### 
clf = DecisionTreeClassifier().fit(X_train, y_train)
print(f'Accuracy of clf training set {clf.score(X_train, y_train)}')
print(f'Accuracy of clf test set {clf.score(X_test, y_test)}')

### K-nearest neighbors ### 
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print(f'Accuracy of knn training set {knn.score(X_train, y_train)}')
print(f'Accuracy of knn test set {knn.score(X_test, y_test)}')


### plot ### 
def plot_fruit_knn(X, y, n_neighbors, weights):
    X_mat = X[['height','width']].values
    #X_mat = X[['mass','color_score']].values
    y_mat = y.values

    #X_mat = scaler.fit_transform(X_mat)

    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF','#AFAFAF'])
    cmap_bold  = ListedColormap(['#FF0000', '#00FF00', '#0000FF','#AFAFAF'])
    
    knn = KNeighborsClassifier(n_neighbors, weights=weights)
    knn.fit(X_mat, y_mat)
    mesh_step_size = .01  # step size in the mesh
    plot_symbol_size = 50
    x_min, x_max = X_mat[:, 0].min() - 1, X_mat[:, 0].max() + 1
    y_min, y_max = X_mat[:, 1].min() - 1, X_mat[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_step_size),
                         np.arange(y_min, y_max, mesh_step_size))
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()]) # np.c_ combine two array into one based on column

    #put the result to a color plot 
    Z = Z.reshape(xx.shape)
    print(Z)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot training points
    plt.scatter(X_mat[:, 0], X_mat[:, 1], s=plot_symbol_size, c=y, cmap=cmap_bold, edgecolor = 'black')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    patch0 = mpatches.Patch(color='#FF0000', label='apple')
    patch1 = mpatches.Patch(color='#00FF00', label='mandarin')
    patch2 = mpatches.Patch(color='#0000FF', label='orange')
    patch3 = mpatches.Patch(color='#AFAFAF', label='lemon')
    plt.legend(handles=[patch0, patch1, patch2, patch3])

    plt.xlabel('height (cm)')
    plt.ylabel('width (cm)')
    plt.title(f"4-Class classification k = {n_neighbors}, weights = {weights}")
    plt.show()

plot_fruit_knn(X, y, 5, 'uniform')

"""
### choose the best n_neighbor
k_range = range(1, 20)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))
plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, scores)
plt.xticks([0,5,10,15,20])

"""
