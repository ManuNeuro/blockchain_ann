# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 09:44:47 2022

@author: Manu
"""

from qiskit import QuantumCircuit, Aer, execute
import matplotlib.pyplot as plt
from qiskit.circuit import Parameter
import numpy as np
from qiskit import IBMQ
from qiskit.tools.monitor import job_monitor
import json

import os
import sys
from pathlib import Path

'''
PATH_ROOT = Path(__file__).parents[1]  # Adjust the number if needed
sys.path.append(str(PATH_ROOT))
from SOURCE_FOLDER.token import token
# IBMQ.save_account(token, overwrite=False)

The first time, you should create an account on IBM-Q: https://www.ibm.com/quantum
Save your API key inside a folder and file token.py.
'''

def random_circuit():
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.measure_all()
    return qc 

def binary_str_to_hex(binary_str):

    # Convert the binary string to an integer
    binary_int = int(binary_str, 2)

    # Convert the integer to a hexadecimal string
    hex_str = hex(binary_int)[2:]

    return hex_str

def generate_private_key(n, backend_name='ibm_kyoto', idx_qubit=0, save=True, save_path=''):
    
    # Run quantum computer
    qc = random_circuit()
    provider = IBMQ.load_account()
    backend = provider.get_backend(backend_name)
    job = execute(qc, backend, shots=n, initial_layout=[idx_qubit],
                  memory=True)
    job_monitor(job)
    
    # Format public key
    binary_str_list = job.result().get_memory()
    public_key_int = np.array(binary_str_list, dtype=int)   
    private_key_str = ''.join(binary_str_list)

    # Metadata
    meta_kwargs = {'backend':backend_name,
                   'size':n,
                   'idx_qubit':idx_qubit}
    # Save private_key and metadata
    dic_private_key = {'private_key(str)':private_key_str,
                       'private_key(int)':public_key_int,
                       'private_key(hex)':binary_str_to_hex(private_key_str),
                       'metadata':meta_kwargs,}
    
    if save:
        path_name = os.path.join(save_path, 'private_key.txt')
        with open(path_name, "w") as fp:
            json.dump(dic_private_key , fp) 
            
    return dic_private_key

def fake_private_key(n, seed=None, periodic=False):
    
    if not periodic:
        if seed is not None:
            np.random.seed(seed)
        private_key = np.random.randint(0, 2, n)
    else:
        private_key = np.zeros(n)
        step = int(n/10)
        private_key[1:n:step] = 1

    private_key_str = np.array(private_key, dtype=str)      
    private_key_str = ''.join(private_key_str)
    public_key_int = np.array(private_key, dtype=int)   
    
    # Save private_key and metadata
    dic_private_key = {'private_key(str)':private_key_str,
                       'private_key(int)':public_key_int,
                       'private_key(hex)':binary_str_to_hex(private_key_str),
                       'metadata':{'backend': 'fake'},
                       }
    
    return dic_private_key