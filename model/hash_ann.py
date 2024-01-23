# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 23:08:54 2024

@author: ManuMan
"""

import numpy as np
import random as rnd
import json 
import os
from whitepaper_p5.model.models_ann import NeuralNetwork

'''
[1] Excitatory/inhibitory balance emerges as a key factor for RBN performance, 
overriding attractor dynamics.
Front. Comput. Neurosci., 09 August 2023
Volume 17 - 2023 | https://doi.org/10.3389/fncom.2023.1223258
'''

hash_ann = {'ann':NeuralNetwork}

def binary_str_to_hex(binary_str):

    # Convert the binary string to an integer
    binary_int = int(binary_str, 2)

    # Convert the integer to a hexadecimal string
    hex_str = hex(binary_int)[2:]

    return hex_str


def calculate_hash_ann(input_data, ann_type='ann', 
                       output_size=512, 
                       seed=None,  
                        **optionalArgs):

    if isinstance(input_data, bytes):
        try:
            input_data = input_data.decode()
        except:
            input_date = str(input_data)
    if isinstance(input_data, str) == True:
        input_data = [ord(c) for c in input_data]
    if isinstance(input_data, list) == True:
        input_data = np.array(input_data, dtype=int)

    input_int = input_data
    input_size = len(input_int)
        
    # Create hash using a neural network
    meta_kwargs = dict(input_dim=input_size,
                            output_dim=optionalArgs.get('output_dim', output_size),
                            hidden_dim=optionalArgs.get('hidden_dim', input_size*2),
                            nb_layer=optionalArgs.get('nb_layer', 100)
                        )
                  
    hash_generator_nn = hash_ann[ann_type](**meta_kwargs)
        
    # Generate weights
    # Seed: for security it's CRITICAL that the seed cannot be found.
    # This is why it is preferable to use the private key as the seed for a seed.
    if seed is None:
        np.random.seed(input_int) # Use private key as seed
        seed = np.random.randint(0, 9, 512) # Generate a new seed with it.
        np.random.seed(seed) # Use this seed for weights generation
        # Notice the format of the private key provided in the seed.
        # private_key_int is an array of int.
    else:
        np.random.seed(seed)
        
    # /!\ It is advice to not change these parameters unless you really know what you are doing:
    # This is because, perfect balance between excitation and inhibition
    # renders maximial chaos in neural network [1].
    mean = optionalArgs.get('mean', 0) # Select a perfect balance between excitation and inhibition.
    std = optionalArgs.get('std', 1) # Sigma can be of any value, as long as it is not zero.
    argWeights = {'mean':mean, 'std':std}
    meta_kwargs.update(argWeights)
    hash_generator_nn.init_weights(**argWeights)
    
    # Format public key
    hash_int = hash_generator_nn.compute_hash(input_int)
    hash_int = np.array(hash_int, dtype=int)   
    hash_str = np.array(hash_int, dtype=str)
    hash_str="".join(hash_str)
    
    hash_hex = binary_str_to_hex(hash_str)

    
    return hash_hex
