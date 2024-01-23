# -*- coding: utf-8 -*-
"""
Created on Sun May 28 19:25:17 2023

@author: ManuMan
"""

import string
import random
import numpy as np
import torch
from torch import nn
from hashlib import sha256

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class PytorchANN(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256, output_dim=256, 
                 nb_layer=3, device=device, training=False):
        super().__init__()
        self.__version__ = 'pytorch-v0.1'
        
        # Flatten the input
        self.flatten = nn.Flatten()
        
        # Creating the architecture
        layers = [nn.Linear(input_dim, hidden_dim, bias=True), nn.ReLU(inplace=False)]
        [layers.extend([nn.Linear(hidden_dim, hidden_dim, bias=True), nn.ReLU(inplace=False)]) for i in range(nb_layer)]
        layers.append(nn.Linear(hidden_dim, output_dim, bias=True))
        self.linear_relu_stack = nn.Sequential(*layers)
        
        # Setting the device
        self.device = device
        self.to(self.device)
        self.linear_relu_stack.to(self.device)
        
        # Values for output
        self.values = torch.tensor([0.0]).to(device)  # Move tensor to the desired device
        
        # No grad
        self.training = training
        if not self.training:
            torch.no_grad() 
        for param in self.parameters():
            param.grad = None
        for param in self.linear_relu_stack.parameters():
            param.grad = None
            
    def compute_hash(self, x, training=False):
        if isinstance(x, np.ndarray) or isinstance(x, list):
            x = torch.from_numpy(np.array(x, dtype=np.float32))
            x = x.unsqueeze(0)
        x = self.flatten(x).to(self.device)
        if training:
            logits = self.linear_relu_stack(x)
        else:
            logits = torch.heaviside(self.linear_relu_stack(x), self.values)
        logits = logits.squeeze(0)
        if not training:
            logits = np.array(logits.tolist(), dtype=int)
        return logits
    
    def init_weights(self, init_func=torch.nn.init.normal_, **kwargs):
        with torch.no_grad():
            for p in self.parameters():
                p = nn.Parameter(init_func(p, **kwargs))

class CpuOptimizedANN(nn.Module):
    def __init__(self, input_dim=500, hidden_dim=100, output_dim=256, 
                 nb_layer=3, cluster_dim=100, device=device):
        super().__init__()
        self.__version__ = 'pytorch-cpu-cluster-v0.1'
        self.input_dim = input_dim
        self.cluster_dim = cluster_dim
        
        # Flatten the input
        self.flatten = nn.Flatten()
        
        # Creating the architecture
        layers = [nn.ReLU(inplace=False), nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=False)]
        [layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=False)]) for i in range(nb_layer)]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.linear_relu_stack = nn.Sequential(*layers)
        
        # Setting the device
        self.device = device
        self.to(self.device)
        self.linear_relu_stack.to(self.device)
        
        # Values for output
        self.values = torch.tensor([0.0]).to(device)  # Move tensor to the desired device
        # Input parameters
        self.weight_input = torch.nn.Parameter(torch.ones(self.cluster_dim,
                                                          self.input_dim)).to(self.device)
        
    def compute_hash(self, x):
        if isinstance(x, np.ndarray) or isinstance(x, list):
            x = torch.from_numpy(np.array(x, dtype=np.float32))
            x = x.unsqueeze(0)
        self.x = x.reshape(self.input_dim, self.cluster_dim)
        inputs = np.einsum('ij,ji->i', self.x, self.weight_input)
        inputs = torch.from_numpy(inputs).unsqueeze(0)
        self.inputs = self.flatten(inputs).to(self.device)
        logits = torch.heaviside(self.linear_relu_stack(self.inputs), self.values)
        logits = logits.squeeze(0)
        logits = np.array(logits.tolist(), dtype=int)
        return logits
    
    def init_weights(self, init_func=torch.nn.init.normal_, mean=0.5, std=10**5):

        # Network parameters    
        for p in self.parameters():
            init_func(p, **kwargs)
        for p in self.linear_relu_stack.parameters():
            init_func(p, **kwargs)
            
        # No grad
        torch.no_grad() 
        self.weight_input.grad = None
        self.weight_input.detach()
        for param in self.parameters():
            param.grad = None
            param.requires_grad = False
        for param in self.linear_relu_stack.parameters():
            param.grad = None
            param.requires_grad = False


class GpuOptimizedANN(nn.Module):
    def __init__(self, input_dim=500, hidden_dim=100, output_dim=256, 
                 nb_layer=3, cluster_dim=100, device=device):
        super().__init__()
        self.__version__ = 'pytorch-gpu-cluster-v0.1'
        self.input_dim = input_dim
        self.cluster_dim = cluster_dim
        self.numberSynapses = cluster_size*input_dim + input_dim*hidden_dim + hidden_dim*output_dim + (hidden_dim**2)*nb_layer
        
        # Flatten the input
        self.flatten = nn.Flatten()
        
        # Creating the architecture
        layers = [nn.ReLU(inplace=False), nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=False)]
        [layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=False)]) for i in range(nb_layer)]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.linear_relu_stack = nn.Sequential(*layers)
        
        # Setting the device
        self.device = device
        self.to(self.device)
        self.linear_relu_stack.to(self.device)
        
        # Values for output
        self.values = torch.tensor([0.0]).to(device)  # Move tensor to the desired device
        # Input parameters
        self.weight_input = torch.nn.Parameter(torch.ones(self.cluster_dim,
                                                          self.input_dim)).to(self.device)
        
    def compute_hash(self, x):
        if isinstance(x, np.ndarray) or isinstance(x, list):
            x = torch.from_numpy(np.array(x, dtype=np.float32))
            x = x.unsqueeze(0)
        x = self.flatten(x).to(self.device)
        self.x = x.reshape(self.input_dim, self.cluster_dim)
        inputs = torch.einsum('ij,ji->i', self.x, self.weight_input)
        self.inputs = inputs.unsqueeze(0)
        logits = torch.heaviside(self.linear_relu_stack(self.inputs), self.values)
        logits = logits.squeeze(0)
        logits = np.array(logits.tolist(), dtype=int)
        return logits
    
    def init_weights(self, init_func=torch.nn.init.normal_, **kwargs):

        # Network parameters    
        for p in self.parameters():
            init_func(p, **kwargs)
        for p in self.linear_relu_stack.parameters():
            init_func(p, **kwargs)
            
        # No grad
        torch.no_grad() 
        self.weight_input.grad = None
        self.weight_input.detach()
        for param in self.parameters():
            param.grad = None
            param.requires_grad = False
        for param in self.linear_relu_stack.parameters():
            param.grad = None
            param.requires_grad = False

class NeuralNetwork:
    def __init__(self, input_dim=256, hidden_dim=256, output_dim=256, nb_layer=3):
        
        # Parameter of the network
        self.nbHiddenLayer = nb_layer
        self.hiddenSize = hidden_dim
        self.inputSize = input_dim
        self.outputSize = output_dim
        self.__version__ = 'NeuralNetwork-v0.1'
        
        # Initializing layers        
        self.inputs = np.zeros(input_dim)
        self.outputs = np.zeros(output_dim)
        self.hiddenLayer = [];
        for l in range(self.nbHiddenLayer):
            self.hiddenLayer.append(np.zeros(self.hiddenSize)) # initialize hidden layer
        
        # Initializing weights
        self.numberSynapses = self.inputSize*self.hiddenSize + self.hiddenSize*self.outputSize + (self.hiddenSize**2)*self.nbHiddenLayer
        self.W = {}
        for l in range(self.nbHiddenLayer+2):
            idLayer = "{0}".format(l)
            if l==0:
                I = self.inputSize
                J = self.hiddenSize
            elif l==self.nbHiddenLayer:
                I = self.hiddenSize
                J = self.outputSize
            else:
                I = self.hiddenSize
                J = I
            self.W.update({idLayer: np.zeros((I, J))})
 
    #########################################################
    # Initialization of network
    #########################################################

    @staticmethod
    def activationFunction(x, option='sigmoid'):
        if option=='sigmoid':
            return 1/(1+np.exp(-(x)))
        elif option=='step':
            output = np.heaviside(x, 0)
            # output[output == 0] = -1
            return  output

    def init_weights(self, vectorWeight=None, **kwargs):
        
        if vectorWeight is None:
            mean = kwargs.get('mean', 0)
            std = kwargs.get('std', 10)
            vectorWeight = np.random.normal(mean, std, self.numberSynapses)
        i=0
        j=-1
        for l in range(self.nbHiddenLayer+2):
            
            # Access weight of given layer
            idLayer = "{0}".format(l)
            W_layer = self.W[idLayer]
            W_size = W_layer.shape
            
            # Get the appropriate indexes of weights 
            j = j+W_size[0]*W_size[1]
            W_toSet = vectorWeight[i:j+1]
            W_toSet = W_toSet.reshape(W_size[0], W_size[1])
            i = j+1
            
            # set weights
            self.W[idLayer] = W_toSet
    
    #########################################################
    # Forward propagation
    #########################################################
    
    def compute_hash(self, X, option='step', debug=False):
        
        # Inputs
        self.inputs = X
        contributions = self.inputs.dot(self.W['0'])
        self.hiddenLayer[0] = self.activationFunction(contributions, option=option)
        if debug:
            print('contribution inputs')
            print(contributions)
            print('hidden layer 0 ------------------')
            print(self.hiddenLayer[0])
        
        # Hidden layers            
        for l in range(1, self.nbHiddenLayer):   
            idLayer = "{0}".format(l)
            contributions = self.hiddenLayer[l-1].dot(self.W[idLayer])
            self.hiddenLayer[l] = self.activationFunction(contributions, option=option)
            if debug:
                print('contribution layer')
                print(contributions)
                print('hidden layer {0} ------------------'.format(l))
                print(self.hiddenLayer[l])
                    
        # output
        if self.nbHiddenLayer==1:
            l=0
        idLayer = "{0}".format(l+1)
        contributions = self.hiddenLayer[l].dot(self.W[idLayer])
        self.outputs = self.activationFunction(contributions, option=option)
        if debug:
            print('contribution last layer')
            print(contributions)
            print('output ------------------')
            print(self.outputs)
        return self.outputs       

class ANN:
    def __init__(self, input_dim, hidden_dim, output_dim,
                 nb_layer, cluster_size):

        # Parameter of the network
        self.nbHiddenLayer = nb_layer
        self.hiddenSize = hidden_dim
        self.inputSize = input_dim
        self.outputSize = output_dim
        self.clusterSize = cluster_size
        self.__version__ = 'NeuralNetwork-v0.2'
        
        # Initializing layers       
        # template = csc_matrix(np.zeros(self.hiddenSize, dtype=np.float16))
        template = np.zeros(self.hiddenSize, dtype=np.float16)
        self.inputs = template.copy()
        self.outputs = template.copy()
        self.hiddenLayer = {};
        for l in range(self.nbHiddenLayer):
            self.hiddenLayer.update({"{0}".format(l):template.copy()}) # initialize hidden layer
        
        # Initializing weights
        self.numberSynapses = cluster_size*input_dim + input_dim*hidden_dim + hidden_dim*output_dim + (hidden_dim**2)*nb_layer
        self.W = {'0': np.zeros((self.clusterSize, self.inputSize))}
        for l in range(nb_layer+2):
            idLayer = "{0}".format(l+1)
            if l==0:
                I = self.inputSize
                J = self.hiddenSize
            elif l==nb_layer:
                I = self.hiddenSize
                J = self.outputSize
            else:
                I = self.hiddenSize
                J = I
            self.W.update({idLayer: np.ones((I, J))})
 
    #########################################################
    # Initialization of network
    #########################################################

    @staticmethod
    def activationFunction(x, option='sigmoid'):
        if option=='sigmoid':
            return 1/(1+np.exp(-(x)))
        elif option=='step':
            output = np.heaviside(x, 0)
            # output[output == 0] = -1
            return  output
        elif option=='relu':
            return x * (x > 0)
        elif option=='identity':
            return x

    def init_weights(self, vectorWeight=None, **kwargs):
        if vectorWeight is None:
            mean = kwargs.get('mean', 0)
            std = kwargs.get('std', 10)
            vectorWeight = np.random.normal(mean, std, self.numberSynapses)
        i=0
        j=-1
        vectorWeight = vectorWeight.astype(np.float16)
        for l in range(self.nbHiddenLayer+2):
            
            # Access weight of given layer
            idLayer = "{0}".format(l)
            W_layer = self.W[idLayer]
            W_size = W_layer.shape
            
            # Get the appropriate indexes of weights 
            j = j+W_size[0]*W_size[1]
            W_toSet = vectorWeight[i:j+1]
            W_toSet = W_toSet.reshape(W_size[0], W_size[1])
            i = j+1
            
            # set weights
            self.W[idLayer] = W_toSet
    
    #########################################################
    # Forward propagation
    #########################################################
    
    def compute_hash(self, X, option='step'):
        
        # Inputs
        self.inputs = X
        self.inputs = self.inputs.reshape(self.inputSize, self.clusterSize)
        
        contributions = np.einsum('ij,ji->i', self.inputs, self.W['0'])
        # contributions = contributions.astype(np.float16)
        self.hiddenLayer['0'] = self.activationFunction(contributions, option=option).astype(int)
        
        # Hidden layers            
        for l in range(1, self.nbHiddenLayer+1):   
            idLayer = "{0}".format(l)
            idPrevLayer = "{0}".format(l-1)
            contributions = self.hiddenLayer[idPrevLayer].dot(self.W[idLayer])
            # contributions = contributions.astype(np.float16)
            self.hiddenLayer[idLayer] = self.activationFunction(contributions, option=option).astype(int)

                    
        # output
        if self.nbHiddenLayer==1:
            l=0
        idLast = "{0}".format(self.nbHiddenLayer+1)
        idLayer = "{0}".format(self.nbHiddenLayer)
        contributions = self.hiddenLayer[idLayer].dot(self.W[idLast])
        # contributions = contributions.astype(np.float16)
        self.outputs = self.activationFunction(contributions, option=option).astype(int)

        return self.outputs       


def hexa_to_bin(ini_string):
    length = 256
    integer = int(ini_string, 16)
    return f'{integer:0>{length}b}'

def sha256_key(input_key_str, option='int'):
    hex_key = sha256(input_key_str.encode('utf-8')).hexdigest()
    key_int = list(hexa_to_bin(hex_key))
    if option == 'int':
        return key_int
    elif option == 'str':
        key_str = np.array(key_int, dtype=str)
        return "".join(key_str)

class SHA256:
    def __init__(self, input_dim=512):
        self._type = 'hash'
        self.input_dim = input_dim
        
    def compute_hash(self, input_str, input_dim=None, option='int'):
        if input_dim is not None:
            self.input_dim = input_dim
        return sha256_key(input_str, option)

