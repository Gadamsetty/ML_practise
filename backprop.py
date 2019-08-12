import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import random as r

#the neural network has L=4(2 hidden) layers
#there will be N=3 input and the output will have K=3 classes

#Each layer has T=5 activation units

#first sep is to initialise the weights with random values
#for an L layer matrix we will have 'L' no.of weight vectors
# for each layer we will also include a bias unit

#initialisation of network
def initial(N,T,K):
	network = list()
	hd_layer = [[r.random()  for i in range(N+1) ]for i in range(T)]
	network.append(hd_layer)
	out_layer = [[r.random() for i in range(T+1)] for i in range(K)]
	network.append(out_layer)
	return network
	
#function to calculate weighted input (z)

def weight_input(wt,aj):
	return np.dot(wt,aj)

def sigmoid(z):
	g = 1/(1+np.exp(-z))
	return g

#function to implement forwrd propagation  

def forward_prop(network,inrow):
	output = list()
	inputs = inrow
	for layer in network:
		out_units = list()
		inputs.insert(0,1)
		for neuron in layer:
			wt_in = weight_input(neuron,inputs)
			g = sigmoid(wt_in)
			out_units.append(g)
			#print(out_units)
		output.append(out_units)
		inputs = out_units
	return output

###########------------BACK PROPAGATION------------#####################

def g_dash_z(aj):
	aj = np.asarray(aj)
	return np.multiply(aj,1-aj)

#function to calculate error at each neuron in each layer

def error_cal(network,outputs,y):
	#error in the final layer
	k = len(outputs)
	l_err = list()
	y_xp = outputs[k-1]
	for i in range(len(y_xp)):
		h = y_xp[i]-y[i]
		l_err.append(h)
		
	er2 = np.asarray(l_err)
	theta = network[1]
	theta = np.asarray(theta)
	e = np.dot(np.transpose(theta),np.transpose(er2))
	#print(e)
	aj = outputs[0]
	g_z_dash = g_dash_z(aj)
	er1 = np.multiply(e,g_z_dash)
	
	return np.array([er1,er2])
#calculation of delta
def delta_cal(outputs,error,x):
	outs = np.asarray(outputs)
	err = np.asarray(error)
	delta = list()
	x.insert(0,1)
	x = np.asarray(x)
	h = np.asarray(outs[0]).reshape(6,1)
	e = err[1].reshape(1,2)
	d2 = h*e
	d2 = d2.transpose()
	e1 = err[0].reshape(1,6)
	x= x.reshape(3,1)
	d1 = x*e1
	d1 = d1.transpose()
	return [d1,d2]
	

netw = initial(2,5,2)
for layer in netw:
	print(layer)
print('\n')
ins = [1,0.7]
outs = forward_prop(netw,ins)
for k in outs:
	print(k)
print('\n')
e = error_cal(netw,outs,[0,1])
print(e)
d = delta_cal(outs,e,[1,0.7])
print(d)

































