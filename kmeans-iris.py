from sklearn.cluster import KMeans
from sklearn import datasets
from itertools import cycle, combinations
import matplotlib.pyplot as pl


iris = datasets.load_iris() # assigns dataset we intend to use
km = KMeans(n_clusters=3) # indicates to the algo how many clusters we wish to
                          # observe, seems to have a max limit once the true number has been found
km.fit(iris.data)         # fits the dataset to the model

predictions = km.predict(iris.data) # calls to find n clusers in the dataset

colors = cycle('rgb') # allows us to choose colors used to represent data points
markers = cycle('^+o') # sets an array to choose shapes used to represent data points
labels = ["Cluster 1","Cluster 2","Cluster 3"] # front end cluster labels
targets = range(len(labels)) # assigns labels to data clusters

feature_index=range(len(iris.feature_names))
feature_names=iris.feature_names # uses the dataset and sets feature_names based on data column names
combs=combinations(feature_index,2) # n indicates how many features we wish to meld in a single graph
                                    # for our use, 2 is probably the best way to viz

f,axarr=pl.subplots(2,3) # Chanes the number of plots printed in the final output, axarr = axis array
axarr_flat=axarr.flat  # links into above comment, I think the f indicates that we are to
                        # be calling on a number, docsets are no help for use of f, axarr

for comb, axflat in zip(combs,axarr_flat):
        for target, color, label, marker in zip(targets,colors,labels,markers):
                feature_index_x=comb[0]
                feature_index_y=comb[1]
                axflat.scatter(iris.data[predictions==target,feature_index_x],
                                iris.data[predictions==target,feature_index_y],c=color,label=label,marker=marker)
                axflat.set_xlabel(feature_names[feature_index_x])
                axflat.set_ylabel(feature_names[feature_index_y])

f.tight_layout()
pl.show()  # the final call, shows us the final graph we we exec script
