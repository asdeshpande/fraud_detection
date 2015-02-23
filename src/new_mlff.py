#!/usr/bin/python
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer, TanhLayer, SoftmaxLayer
from pybrain.structure import FullConnection
from pybrain.datasets import ClassificationDataSet, SupervisedDataSet, UnsupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.unsupervised.trainers.deepbelief import DeepBeliefTrainer
from pybrain.supervised.trainers import Trainer
from pybrain.structure.networks.rbm import Rbm
from pybrain.unsupervised.trainers.rbm import (RbmGibbsTrainerConfig,
                                               RbmBernoulliTrainer)
import csv
import numpy

# set up a basic feed forward network
net = FeedForwardNetwork()
ds = ClassificationDataSet(9,1,nb_classes=2, class_labels=['FRAUD', 'N'])
temp_ds = UnsupervisedDataSet(9)

# define 3 layers
inLayer = LinearLayer(9, "visible")
hiddenLayer = SigmoidLayer(16)
outLayer = LinearLayer(1)

# add layers to network
net.addInputModule(inLayer)
net.addModule(hiddenLayer)
net.addOutputModule(outLayer)

# define connections between layers
in_to_hidden = FullConnection(inLayer, hiddenLayer)
hidden_to_out = FullConnection(hiddenLayer, outLayer)

# add connections to the network
net.addConnection(in_to_hidden)
net.addConnection(hidden_to_out)

# Sort modules topologically and initialize
net.sortModules()

with open('vsample.csv','rb') as f:
    reader = csv.reader(f)
    for row in reader:
        d_input = map(float,row[1:10])
        output = map(float, row[10])
        n_input = d_input/numpy.linalg.norm(d_input)
        ds.addSample(d_input,output)
        temp_ds.addSample(d_input)

#print ds       
cfg = RbmGibbsTrainerConfig()
cfg.maxIter = 3
rbm = Rbm.fromDims(9,5)
trainer = BackpropTrainer(net, dataset=ds, learningrate= 0.001, weightdecay=0.01, verbose=True)
#trainer = DeepBeliefTrainer(net, dataset=temp_ds)
#trainer = RbmBernoulliTrainer(rbm, temp_ds, cfg)
for i in range(30):
	trainer.trainEpochs(30)
	
print 'Expected:1 [FRAUD]     ', net.activate([49,2.6,0.98,4.3,1.48,10,2.5,6,67]) 
print 'Expected:0 [NOT FRAUD] ', net.activate([78,5,4.4,4.5,2.99,3,1.3,10,59])
print 'Expected:1 [FRAUD]     ', net.activate([57,2,0.1,1.15,0.47,7,1.8,6,73])
print 'Expected:0 [NOT FRAUD] ', net.activate([65,3,11.1,1.8,0.6,4,4,4.5,90])
print 'Expected:1 [FRAUD]     ', net.activate([55,2,0.23,3.2,0.55,9,1.9,5.5,60])
print 'Expected:0 [NOT FRAUD] ', net.activate([39,5,0.07,0.5,0.17,3,3.8,3,32])
print 'Expected:1 [FRAUD]     ', net.activate([63,2.5,0.25,1.23,0.3,7,1.45,4.75,35])   

