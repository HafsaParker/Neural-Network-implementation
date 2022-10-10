#!/usr/bin/env python
# coding: utf-8

# In[7]:


import sys
import numpy as np
import matplotlib


# In[2]:


##version checking
print("Python version", sys.version)
print("Numpy version", np.__version__)
print("matplotlib",matplotlib.__version__)


# In[5]:


##Lets code Neuron
##lets assume we have 3 neurons, these 3 neurons are inputs from some output neurons.
##lets take the input first.
Input = [1,2,3,2.5]
## Now every unique input will have a weight assocuated with them.
weights = [0.2,0.8,-0.5,1.0]
##Now every unique neuron have a unique bias.
bias  = 2
##now the first thing neuron will do is to add up all input times thw weight plus bias
output = Input[0]*weights[0]+Input[1]*weights[1]+Input[2]*weights[2]+Input[3]*weights[3]+bias
print(output)


# In[6]:


##coding 3 neurons with 4 inputs
Input = [1,2,3,2.5]
#Now for 3 neurons will have 3 different wieght
weights1 = [0.2,0.8,-0.5,1.0]
weights2 = [0.5,-0.91,0.26,-0.5]
weights3 = [-0.26,-0.27,0.17,0.87]
bias1  = 2
bias2  = 3
bias3  = 0.5
##becuase output can be an input for next neutron
output = [Input[0]*weights1[0]+Input[1]*weights1[1]+Input[2]*weights1[2]+Input[3]*weights1[3]+bias1,
         Input[0]*weights2[0]+Input[1]*weights2[1]+Input[2]*weights2[2]+Input[3]*weights2[3]+bias2,
         Input[0]*weights3[0]+Input[1]*weights3[1]+Input[2]*weights3[2]+Input[3]*weights3[3]+bias3]
print(output)


# In[5]:


##coding more dynamically
Input = [1,2,3,2.5]
weights = [[0.2,0.8,-0.5,1.0],[0.5,-0.91,0.26,-0.5], [-0.26,-0.27,0.17,0.87]]
bias = [2,3,0.5]
output_layer = []

for neuron_weight,neuron_bias in zip(weights,bias):
    neuron_output = 0
    for n_input,n_weight in zip(Input,neuron_weight):
        neuron_output += n_weight*n_input
    neuron_output += neuron_bias
    output_layer.append(neuron_output)
print(output_layer)
        
    

##understanding shape()--> demostrate the size of dimension 
## array =[1,2,3,4,5] shape: (4,)   array = [[1,2,3],[3,4,3]]  shape: (2,3) because 2D array
## what is tensor? its an object that can be represented as array
## --------------THE DOT PRODUCT (with layers of neurons)------------------
# In[10]:


weights = [[0.2,0.8,-0.5,1.0],[0.5,-0.91,0.26,-0.5], [-0.26,-0.27,0.17,0.87]]
Input = [1,2,3,2.5]
output = np.dot(weights,Input)+bias
print(output)


# In[ ]:




