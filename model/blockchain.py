# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 10:56:51 2024

@author: ManuMan
"""

import pickle
import time
import hashlib
import importlib.util
import hashlib

from blockchain_ann.model.transaction import TransactionPool
from blockchain_ann.model.mining_ann import MiningANN

copy = lambda obj: pickle.loads(pickle.dumps(obj))

def compute_hash(file_path):
    with open(file_path, 'rb') as f:
        data = f.read()    
    return hashlib.sha256(data).hexdigest()

def load_class(file_path, class_name):
    module_name = file_path.split('/')[-1].rstrip('.py')  # Remove .py from file_path to get module name
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)

class Block:
    def __init__(self, index, previous_hash, timestamp, transactions, hash=None, difficulty=None):
        self.index = index
        self.previous_hash = previous_hash
        self.timestamp = timestamp
        self.transactions = copy(transactions)  # Added transactions
        self.difficulty = difficulty
        self.hash_model = None
        self.hash_file = None
        
        if hash is None:
            self.calculate_hash()
        else:
            self.hash = hash
    
    def calculate_hash(self):
        value = str(self.index)
        value += str(self.previous_hash) 
        value += str(self.timestamp) 
        value += str(self.transactions) 
        value += str(self.difficulty)
        value += str(self.hash_model)
        value += str(self.hash_file)
        self.hash = hashlib.sha256(value.encode('utf-8')).hexdigest()  
        return self.hash 

    def set_hash_mined(self, hash_model, hash_file):
        self.hash_model = hash_model
        self.hash_file = hash_file
        self.calculate_hash()
    
    def __str__(self):
        text = f"Block #{self.index}\nHash: {self.hash}\n"
        text += str(self.transactions)
        return text

class Blockchain:
    def __init__(self, mining_chain):
        self.pending_transactions = TransactionPool() 
        self.chain = [] # Ordered list of the hashs
        self.blocks = {} # Storing of the blocks
        
        # Private attributes
        self.__mining_reward = 10  # For example
        self.__path_mining = './model/mining_ann.py'
        self.__hash_file_mining = compute_hash(self.__path_mining)

        # Add the genesis block        
        genesis_block = self.__create_genesis_block(mining_chain)
        self.__add_new_block(genesis_block)
        
    # Private method
    def __create_genesis_block(self, mining_chain):
        timestamp = int(time.time())
        self.pending_transactions.add_transaction("0",  "0", 0)
        genesis_block = Block(0, "0", timestamp, self.pending_transactions)
        self.pending_transactions.clear()
        genesis_block = mining_chain.synchronize_block(genesis_block)
        return genesis_block

    # Private method
    def __create_new_block(self, previous_block, transactions, difficulty):
        index = previous_block.index + 1
        timestamp = int(time.time())
        return Block(index, previous_block.hash, timestamp, transactions, difficulty=difficulty)
    
    # Private method
    def __add_new_block(self, new_block):
        # Add the block to the chain
        self.chain.append(new_block.hash)
        self.blocks.update({new_block.hash: new_block})
        self.block_height = len(self.chain) - 1
    
    def get_latest_block(self):
        latest_block_hash = self.chain[-1]
        return self.blocks[latest_block_hash]
    
    def mine_new_block(self, transactions, difficulty, mining_chain, wallets, miner_public_key):
        
        # Add the transactions to the list of pending transactions
        self.pending_transactions.add_transactions(transactions)
            
        # Add reward to transactions
        self.__reward_miner(miner_public_key)
        
        # Create the new block
        new_block = self.__create_new_block(self.get_latest_block(), self.pending_transactions, difficulty)

        ## Mining  
        # Check authenticity of the model 
        if compute_hash(self.__path_mining) == self.__hash_file_mining:
            MiningANN = load_class(self.__path_mining, 'MiningANN')
        else:
            print('The MiningANN class has been tampered with, block not created')
            return False
        
        # Perform mining
        classifier = MiningANN()
        accuracy = 0
        while accuracy < difficulty / 100:
            classifier.train()
            print(f'Difficulty is set to {difficulty}%')
            accuracy = classifier.test()
            
        # Synchronize blockchain and mining chain
        new_block = mining_chain.synchronize_block(new_block, classifier)
        self.__add_new_block(new_block) 
        print(f'New block #{new_block.index} is mined!')
        
        # Execute the transactions
        self.pending_transactions.execute_transaction(wallets)
        reward_transaction = self.pending_transactions.transactions[-1]
        wallets[miner_public_key].add_UTXO(reward_transaction)
        
        # Clear the pending transactions
        self.pending_transactions.clear()
        return True
    
    def __reward_miner(self, public_key): 
        # Add the reward transaction to the new block
        self.pending_transactions.add_transaction('0', public_key, self.__mining_reward)
    
def validate_chain(blockchain, mining_chain, wallets):
    for i in range(1, len(blockchain.chain)):
        current_block = blockchain.blocks[blockchain.chain[i]]
        previous_block = blockchain.blocks[blockchain.chain[i - 1]]
        
        # Check mining gives the correct accuracy
        mining_model = mining_chain.validate_model(i)
        if mining_model.test() < current_block.difficulty / 100:
            print(f'Invalid mining model #{i}')
            return False
        
        # Check that the hash of the block is correct
        if current_block.hash != current_block.calculate_hash():
            print(f'Invalid block #{i}')
            return False

        # Check that the previous hash is correct
        if current_block.previous_hash != previous_block.hash:
            print('Invalid chai n#{i}')
            return False
        
        # Validate transactions
        if not current_block.transactions.validate_transactions(wallets):
            print(f'Invalid transaction #{i}')
            return False 
        
    return True