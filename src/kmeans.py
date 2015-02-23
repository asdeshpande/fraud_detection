#!/usr/bin/python
from pylab import plot,show,savefig
from numpy import vstack,array
from numpy.random import rand
from scipy.cluster.vq import kmeans,vq
import numpy, csv
from pybrain.datasets import ClassificationDataSet, SupervisedDataSet

ds = ClassificationDataSet(9,1,nb_classes=2, class_labels=['FRAUD', 'N'])
temp_ds = SupervisedDataSet(9,1)
# data generation
#data = vstack((rand(150,2) + array([.5,.5]),rand(150,2)))
with open('vsample.csv','rb') as f:
    reader = csv.reader(f)
    for row in reader:
        names = row[0]
        d_input = map(float,row[1:10])
        output = map(float, row[10])
        n_input = d_input/numpy.linalg.norm(d_input)
        ds.addSample(d_input,output)
        temp_ds.addSample(n_input,output)
'''
t_data = []
for each in raw_data:
    row = each/numpy.linalg.norm(each)
    t_data.append(row)
'''
data = numpy.asarray(temp_ds['input']) 

# computing K-Means with K = 2 (2 clusters)
centroids,_ = kmeans(data,2)
# assign each sample to a cluster
idx,_ = vq(data,centroids)

# some plotting using numpy's logical indexing
plot(data[idx==0,0],data[idx==0,1],'ob',
     data[idx==1,0],data[idx==1,1],'or',label=names,markersize=12)
plot(centroids[:,0],centroids[:,1],'sg',markersize=8)
savefig("kmeans100.png")
show()
