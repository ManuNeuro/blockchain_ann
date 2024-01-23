# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 10:08:11 2022

@author: Manu
"""

import numpy as np
import random as rnd
import json 
import os
from blockchain_ann.model.models_ann import PytorchANN, NeuralNetwork

'''
[1] Excitatory/inhibitory balance emerges as a key factor for RBN performance, 
overriding attractor dynamics.
Front. Comput. Neurosci., 09 August 2023
Volume 17 - 2023 | https://doi.org/10.3389/fncom.2023.1223258
'''

hash_ann = {'pyann':PytorchANN,
            'ann':NeuralNetwork}

def binary_str_to_hex(binary_str):

    # Convert the binary string to an integer
    binary_int = int(binary_str, 2)

    # Convert the integer to a hexadecimal string
    hex_str = hex(binary_int)[2:]

    return hex_str

def generate_public_key(private_key, ann_type='ann', 
                        size_public_key=None, 
                        seed=None, 
                        key_generator_nn=None, 
                        save=True, save_path='', 
                        **optionalArgs):
    
    # Convert the private_key in int    
    private_key_int = np.array(list(private_key), dtype=int)
    
    meta_kwargs = {}
    # Generate network (if needed)
    if key_generator_nn is None:
        # Size of input and output of the network
        size_private_key = len(private_key)    
        if size_public_key is None:
            size_public_key = size_private_key
        # Create key_generator using NeuralNetwork
        meta_kwargs.update(dict(input_dim=size_private_key,
                                output_dim=optionalArgs.get('output_dim', size_public_key),
                                hidden_dim=optionalArgs.get('hidden_dim', size_public_key),
                                nb_layer=optionalArgs.get('nb_layer', 50)),
                      )
        key_generator_nn = hash_ann[ann_type](**meta_kwargs)
        meta_kwargs.update(dict({'key_generator':key_generator_nn.__version__}))
    else:
        # If the key_generator is provided, we strongly recommend storing the architecture parameters.
        # Store it in `custom` inside the optionalArgs dic, to be able to regenerate the public_key from the private_key
        customArchitecture = optionalArgs.get('custom', dict(custom="WARNING: No information provided"))
        meta_kwargs.update(customArchitecture)
        
    # Generate weights
    # Seed: for security it's CRITICAL that the seed cannot be found.
    # This is why it is preferable to use the private key as the seed for a seed.
    if seed is None:
        np.random.seed(private_key_int) # Use private key as seed
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
    key_generator_nn.init_weights(**argWeights)
    
    # Format public key
    public_key_int = key_generator_nn.compute_hash(private_key_int)
    public_key_int = np.array(public_key_int, dtype=int) 
    public_key_str = np.array(public_key_int, dtype=str)
    public_key_str="".join(public_key_str)
    
    # Save public_key and metadata
    dic_public_key = {'public_key(str)':public_key_str,
                       'public_key(int)':public_key_int,
                       'public_key(hex)':binary_str_to_hex(public_key_str),
                       'metadata':meta_kwargs,}
    
    if save:
        path_name = os.path.join(save_path, "public_key.txt")
        with open(path_name, "w") as fp:
            json.dump(dic_public_key , fp) 
    
    return dic_public_key


def bit_flip(array_bin, nb_flip):
    array_bin = array_bin.copy()
    indexes = np.arange(len(array_bin))
    indexes_flip = rnd.sample(list(indexes), nb_flip)    
    array_bin[indexes_flip] = 1 - array_bin[indexes_flip]
    return array_bin

