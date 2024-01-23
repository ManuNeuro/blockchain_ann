# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 18:57:45 2024

@author: ManuMan
"""

import hashlib
import time 
import sys
from pathlib import Path
PATH_ROOT = Path(__file__).parents[1]  # Adjust the number if needed
sys.path.append(str(PATH_ROOT))

# The model and key generator
from whitepaper_p5.model.public_key_generator import generate_public_key, bit_flip
from whitepaper_p5.model.private_key_generator import generate_private_key

# %% Block

class Block:
    def __init__(self, index, previous_hash, timestamp, data, hash):
        self.index = index
        self.previous_hash = previous_hash
        self.timestamp = timestamp
        self.data = data
        self.hash = hash

def calculate_hash(index, previous_hash, timestamp, data):
    value = str(index) + str(previous_hash) + str(timestamp) + str(data)
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def create_genesis_block():
    return Block(0, "0", int(time.time()), "Genesis Block", calculate_hash(0, "0", int(time.time()), "Genesis Block"))


def create_new_block(previous_block, data):
    index = previous_block.index + 1
    timestamp = int(time.time())
    hash = calculate_hash(index, previous_block.hash, timestamp, data)
    return Block(index, previous_block.hash, timestamp, data, hash)

# Création du bloc de genèse
genesis_block = create_genesis_block()
print("Genesis Block Hash: ", genesis_block.hash)

# Création d'un nouveau bloc
new_block = create_new_block(genesis_block, "Some data for the new block")
print("New Block Hash: ", new_block.hash)

# %% Blockchain v0.1

class Blockchain:
    def __init__(self):
        # Initialize the blockchain with the genesis block as the only item in the chain
        self.chain = [self.create_genesis_block()]

    def create_genesis_block(self):
        # Create the genesis block, which is the first block in the blockchain
        return Block(0, "0", int(time.time()), "Genesis Block", calculate_hash(0, "0", int(time.time()), "Genesis Block"))

    def get_latest_block(self):
        # Return the last block in the chain, which is useful for creating new blocks
        return self.chain[-1]

    def add_new_block(self, data):
        # Create a new block with the provided data and add it to the chain
        new_block = create_new_block(self.get_latest_block(), data)
        self.chain.append(new_block)
        
# Create a new blockchain
blockchain = Blockchain()

# Add new blocks
blockchain.add_new_block("Some data for the first block")
blockchain.add_new_block("Some data for the second block")

# Print the blockchain
for block in blockchain.chain:
    print("Block #", block.index, " Hash: ", block.hash)
    
# %% Blockchain v0.2

class Blockchain:
    def __init__(self):
        self.chain = [self.create_genesis_block().hash]
        self.blocks = {self.chain[0]: self.create_genesis_block()}

    def create_genesis_block(self):
        genesis_block = Block(0, "0", int(time.time()), "Genesis Block", calculate_hash(0, "0", int(time.time()), "Genesis Block"))
        return genesis_block

    def get_latest_block(self):
        latest_block_hash = self.chain[-1]
        return self.blocks[latest_block_hash]

    def add_new_block(self, data):
        new_block = create_new_block(self.get_latest_block(), data)
        self.chain.append(new_block.hash)
        self.blocks[new_block.hash] = new_block

# Create a new blockchain
blockchain = Blockchain()

# Add new blocks
blockchain.add_new_block("Some data for the first block")
blockchain.add_new_block("Some data for the second block")

# Print the blockchain
for hash in blockchain.chain:
    block = blockchain.blocks[hash]
    print("Block #", block.index, " Hash: ", block.hash)
    
# %% Blockchain v0.3
class Block:
    def __init__(self, index, previous_hash, timestamp, data, hash, nonce=0):
        self.index = index
        self.previous_hash = previous_hash
        self.timestamp = timestamp
        self.data = data
        self.hash = hash
        self.nonce = nonce

def calculate_hash(index, previous_hash, timestamp, data, nonce):
    value = str(index) + str(previous_hash) + str(timestamp) + str(data) + str(nonce)
    return hashlib.sha256(value.encode('utf-8')).hexdigest()    

class Blockchain:
    def __init__(self):
        self.chain = [self.__create_genesis_block().hash]
        self.blocks = {self.chain[0]: self.__create_genesis_block()}
    
    # Private method
    def __create_genesis_block(self):
        genesis_block = Block(0, "0", int(time.time()), "Genesis Block", calculate_hash(0, "0", int(time.time()), "Genesis Block", 0))
        return genesis_block
    
    # Private method
    def __create_new_block(self, previous_block, data, nonce=0):
        index = previous_block.index + 1
        timestamp = int(time.time())
        hash = calculate_hash(index, previous_block.hash, timestamp, data, nonce)
        return Block(index, previous_block.hash, timestamp, data, hash, nonce)
    
    # Private method
    def __add_new_block(self, new_block):
        self.chain.append(new_block.hash)
        self.blocks.update({new_block.hash: new_block})
    
    def get_latest_block(self):
        latest_block_hash = self.chain[-1]
        return self.blocks[latest_block_hash]
    
    def mine_new_block(self, data, difficulty):
        new_block = self.__create_new_block(self.get_latest_block(), data)

        while new_block.hash[:difficulty] != "0" * difficulty:
            new_block.nonce += 1
            new_block.hash = calculate_hash(new_block.index, new_block.previous_hash, new_block.timestamp, new_block.data, new_block.nonce)

        self.__add_new_block(new_block)
        print(f'New block #{new_block.index} is mined!')
        
# Create a new blockchain
blockchain = Blockchain()

# Add new blocks
blockchain.mine_new_block("Some data for the first block", 6)
blockchain.mine_new_block("Some data for the second block", 6)

# Print the blockchain
for hash in blockchain.chain:
    block = blockchain.blocks[hash]
    print("Block #", block.index, " Hash: ", block.hash)
    
# %% Blockchain V0.4

class Transaction:
    # This is a simple example, a real transaction would be more complex
    def __init__(self, sender, receiver, amount):
        self.sender = sender
        self.receiver = receiver
        self.amount = amount
    
    def __str__(self):
        return f"{self.sender} -> {self.receiver}: {self.amount}"

    def is_valid(self):
        # Here we would check that the transactions are valid
        # For example, check that the sender has enough balance for the transaction
        return True

class Block:
    def __init__(self, index, previous_hash, timestamp, transactions, hash, nonce=0):
        self.index = index
        self.previous_hash = previous_hash
        self.timestamp = timestamp
        self.transactions = transactions  # Added transactions
        self.hash = hash
        self.nonce = nonce

    def validate_transactions(self):
        # Check that all transactions are valid
        for transaction in self.transactions:
            if not transaction.is_valid():
                return False
        return True


def calculate_hash(index, previous_hash, timestamp, data, nonce):
    value = str(index) + str(previous_hash) + str(timestamp) + str(data) + str(nonce)
    return hashlib.sha256(value.encode('utf-8')).hexdigest()    

class Blockchain:
    def __init__(self):
        self.chain = [self.__create_genesis_block().hash]
        self.blocks = {self.chain[0]: self.__create_genesis_block()}
    
    # Private method
    def __create_genesis_block(self):
        genesis_block = Block(0, "0", int(time.time()), "Genesis Block", calculate_hash(0, "0", int(time.time()), "Genesis Block", nonce=0))
        return genesis_block
    
    # Private method
    def __create_new_block(self, previous_block, transactions, nonce=0):
        index = previous_block.index + 1
        timestamp = int(time.time())
        hash = calculate_hash(index, previous_block.hash, timestamp, transactions, nonce)
        return Block(index, previous_block.hash, timestamp, transactions, hash, nonce)
    
    # Private method
    def __add_new_block(self, new_block):
        self.chain.append(new_block.hash)
        self.blocks[new_block.hash] = new_block
    
    def get_latest_block(self):
        latest_block_hash = self.chain[-1]
        return self.blocks[latest_block_hash]
    
    def mine_new_block(self, transactions, difficulty):
        new_block = self.__create_new_block(self.get_latest_block(), transactions)

        # Validate the transactions
        if not new_block.validate_transactions():
            print("Invalid transactions, block not created")
            return False
         
        while new_block.hash[:difficulty] != "0" * difficulty:
            new_block.nonce += 1
            new_block.hash = calculate_hash(new_block.index, new_block.previous_hash, new_block.timestamp, new_block.transactions, new_block.nonce)

        self.__add_new_block(new_block)
        print(f'New block #{new_block.index} is mined!')
    
    def reward_miner(self, miner_wallet):
        # Create a new transaction that adds the reward to the miner's wallet
        reward_transaction = Transaction('0', miner_wallet.public_key, self.mining_reward)
        # Add the reward transaction to the new block
        self.pending_transactions.append(reward_transaction)
    
    def validate_chain(self):
        for i in range(1, len(self.chain)):
            current_block = self.blocks[self.chain[i]]
            previous_block = self.blocks[self.chain[i - 1]]

            # Check that the hash of the block is correct
            if current_block.hash != calculate_hash(current_block.index, current_block.previous_hash, current_block.timestamp, current_block.transactions, current_block.nonce):
                return False

            # Check that the previous hash is correct
            if current_block.previous_hash != previous_block.hash:
                return False
            
            if not current_block.validate_transactions():
                return False

        return True

    

# Create a new blockchain
blockchain = Blockchain()

# Add new blocks
transactions = [Transaction("Alice", "Bob", 50), Transaction("Bob", "Charlie", 25)]
blockchain.mine_new_block(transactions, 5)
transactions = [Transaction("Alice", "Eve", 150), Transaction("Alice", "Charlie", 12)]
blockchain.mine_new_block(transactions, 5)

# Print the blockchain
for hash in blockchain.chain:
    block = blockchain.blocks[hash]
    print("Block #", block.index, " Hash: ", block.hash)

print('Blockchain validation:', blockchain.validate_chain())

# %% Wallet v0.1

import binascii

class Wallet:
    def __init__(self):
        self.private_key, self.public_key = self.generate_keys()
        self.UTXOs = {}  # UTXOs owned by this wallet
        self.value = 0.0  # Value of UTXOs owned by this wallet

    def generate_keys(self):
        # Generate a new private/public key pair
        private_key = generate_private_key(1024, save=False)
        print(private_key['private_key(int)'])
        public_key = generate_public_key(private_key['private_key(int)'], save=False)
        return (private_key['private_key(hex)'],
                public_key['public_key(hex)'])

    def update_value(self):
        # Update the value of UTXOs owned by this wallet
        self.value = sum(UTXO.amount for UTXO in self.UTXOs.values())

    def add_UTXO(self, transaction_id, output_index, amount):
            # Create a new UTXO from the transaction
            utxo = UTXO(transaction_id, output_index, self.public_key, amount)
    
            # Add the UTXO to the wallet
            self.UTXOs[utxo.transaction_id] = utxo
    
            # Update the value of the wallet
            self.update_value()

class UTXO:
    def __init__(self, transaction_id, output_index, owner, amount):
        self.transaction_id = transaction_id
        self.output_index = output_index
        self.owner = owner
        self.amount = amount
        
wallet1 = Wallet()
print("Nouveau portefeuille créé !")
print("Clé privée :", wallet1.private_key)
print("Clé publique :", wallet1.public_key)
print("Valeur :", wallet1.value)    
    
transaction1 = Transaction(wallet1.public_key, 'receiver_public_key', 10)

wallet1.add_UTXO(transaction1, 0, 10)


# %% Blockchain v0.5

class Wallet:
    def __init__(self):
        self.private_key, self.public_key = self.generate_keys()
        self.UTXOs = {}  # UTXOs owned by this wallet
        self.value = 0.0  # Value of UTXOs owned by this wallet

    def generate_keys(self):
        # Generate a new private/public key pair
        private_key = generate_private_key(1024, save=False)
        print(private_key['private_key(int)'])
        public_key = generate_public_key(private_key['private_key(int)'], save=False)
        return (private_key['private_key(hex)'],
                public_key['public_key(hex)'])

    def update_value(self):
        # Update the value of UTXOs owned by this wallet
        self.value = sum(UTXO.amount for UTXO in self.UTXOs.values())

    def add_UTXO(self, transaction):
            # Create a new UTXO from the transaction
            utxo = UTXO(transaction.id, transaction.output_index, self.public_key, transaction.amount)
    
            # Add the UTXO to the wallet
            self.UTXOs[utxo.transaction_id] = utxo
    
            # Update the value of the wallet
            self.update_value()

class UTXO:
    def __init__(self, transaction_id, output_index, owner, amount):
        self.transaction_id = transaction_id
        self.output_index = output_index
        self.owner = owner
        self.amount = amount
        

class Block:
    def __init__(self, index, previous_hash, timestamp, transactions, hash, nonce=0):
        self.index = index
        self.previous_hash = previous_hash
        self.timestamp = timestamp
        self.transactions = transactions  # Added transactions
        self.hash = hash
        self.nonce = nonce
        
    def validate_transactions(self):
        # Check that all transactions are valid
        for transaction in self.transactions:
            if not transaction.is_valid():
                return False
        return True

    def __str__(self):
        text = f"Block #{self.index}\nHash: {self.hash}\n"
        for transaction in transactions:
            text += str(transaction) + '\n'
            text += '---------------------\n'
        return text

class Blockchain:
    def __init__(self):
        self.chain = [self.__create_genesis_block().hash]
        self.blocks = {self.chain[0]: self.__create_genesis_block()}
        self.pending_transactions = []  # New attribute
        self.__mining_reward = 10  # For example
        
    # Private method
    def __create_genesis_block(self):
        genesis_block = Block(0, "0", int(time.time()), "Genesis Block", calculate_hash(0, "0", int(time.time()), "Genesis Block", nonce=0))
        return genesis_block
    
    # Private method
    def __create_new_block(self, previous_block, transactions, nonce=0):
        index = previous_block.index + 1
        timestamp = int(time.time())
        hash = calculate_hash(index, previous_block.hash, timestamp, transactions, nonce)
        return Block(index, previous_block.hash, timestamp, transactions, hash, nonce)
    
    # Private method
    def __add_new_block(self, new_block):
        self.chain.append(new_block.hash)
        self.blocks[new_block.hash] = new_block
    
    def get_latest_block(self):
        latest_block_hash = self.chain[-1]
        return self.blocks[latest_block_hash]
    
    def mine_new_block(self, transactions, difficulty, miner_wallet):
        
        # Add the transactions to the list of pending transactions
        self.pending_transactions.extend(transactions)
            
        # Add reward to transactions
        self.reward_miner(miner_wallet)
        
        for trans in self.pending_transactions:
            print(trans)
        
        new_block = self.__create_new_block(self.get_latest_block(), self.pending_transactions)

        # Validate the transactions
        if not new_block.validate_transactions():
            print("Invalid transactions, block not created")
            return False
    
        # Mining         
        while new_block.hash[:difficulty] != "0" * difficulty:
            new_block.nonce += 1
            new_block.hash = calculate_hash(new_block.index, new_block.previous_hash, new_block.timestamp, new_block.transactions, new_block.nonce)
        
        self.__add_new_block(new_block)
        print(f'New block #{new_block.index} is mined!')
        
        # Reward the miner        
        reward_transaction = self.pending_transactions[-1]  # The reward transaction is the last one in the list
        miner_wallet.add_UTXO(reward_transaction)

        # Clear the pending transactions
        self.pending_transactions = []
    
    def reward_miner(self, miner_wallet):
        # Create a new transaction that adds the reward to the miner's wallet
        reward_transaction = Transaction('0', miner_wallet.public_key, self.__mining_reward)

        # Add the reward transaction to the new block
        self.pending_transactions.append(reward_transaction)
    
    def validate_chain(self):
        for i in range(1, len(self.chain)):
            current_block = self.blocks[self.chain[i]]
            previous_block = self.blocks[self.chain[i - 1]]

            # Check that the hash of the block is correct
            if current_block.hash != calculate_hash(current_block.index, current_block.previous_hash, current_block.timestamp, current_block.transactions, current_block.nonce):
                return False

            # Check that the previous hash is correct
            if current_block.previous_hash != previous_block.hash:
                return False
            
            if not current_block.validate_transactions():
                return False

        return True

# Create a new blockchain
blockchain = Blockchain()

# Add new blocks
transactions = [Transaction("Alice", "Bob", 50), Transaction("Bob", "Charlie", 25)]
blockchain.mine_new_block(transactions, 5, wallet1)
transactions = [Transaction("Alice", "Eve", 150), Transaction("Alice", "Charlie", 12)]
blockchain.mine_new_block(transactions, 5, wallet1)

# Print the blockchain
for hash in blockchain.chain:
    block = blockchain.blocks[hash]
    print(block)

print('Blockchain validation:', blockchain.validate_chain())

# %% Blockchain V0.6

import uuid
import pickle

copy = lambda obj: pickle.loads(pickle.dumps(obj))

def calculate_hash(index, previous_hash, timestamp, data, nonce):
    value = str(index) + str(previous_hash) + str(timestamp) + str(data) + str(nonce)
    return hashlib.sha256(value.encode('utf-8')).hexdigest()  

class Wallet:
    def __init__(self):
        self.private_key, self.public_key = self.generate_keys()
        self.UTXOs = {}  # UTXOs owned by this wallet
        self.value = 0.0  # Value of UTXOs owned by this wallet

    def generate_keys(self):
        # Generate a new private/public key pair
        private_key = generate_private_key(1024, save=False)
        print(private_key['private_key(int)'])
        public_key = generate_public_key(private_key['private_key(int)'], save=False)
        return (private_key['private_key(hex)'],
                public_key['public_key(hex)'])

    def update_value(self):
        # Update the value of UTXOs owned by this wallet
        self.value = sum(UTXO.amount for UTXO in self.UTXOs.values())

    def add_UTXO(self, transaction):
            # Create a new UTXO from the transaction
            utxo = UTXO(transaction.id, transaction.output_index, self.public_key, transaction.amount)
    
            # Add the UTXO to the wallet
            self.UTXOs[utxo.transaction_id] = utxo
    
            # Update the value of the wallet
            self.update_value()

class UTXO:
    def __init__(self, transaction_id, output_index, owner, amount):
        self.transaction_id = transaction_id
        self.output_index = output_index
        self.owner = owner
        self.amount = amount

class Block:
    def __init__(self, index, previous_hash, timestamp, transactions, hash, nonce=0):
        self.index = index
        self.previous_hash = previous_hash
        self.timestamp = timestamp
        self.transactions = copy(transactions)  # Added transactions
        self.hash = hash
        self.nonce = nonce
        
    def __str__(self):
        text = f"Block #{self.index}\nHash: {self.hash}\n"
        text += str(self.transactions)
        return text

class Transaction:
    # This is a simple example, a real transaction would be more complex
    def __init__(self, sender, receiver, amount, transaction_id=None):
        self.id = transaction_id
        self.sender = sender
        self.receiver = receiver
        self.amount = amount
    
    def __str__(self):
        return f"{self.sender} -> {self.receiver}: {self.amount}"

    def is_valid(self):
        # Here we would check that the transactions are valid
        # For example, check that the sender has enough balance for the transaction
        return True
    
class TransactionPool:
    def __init__(self):
        self.transactions = []

    def add_transaction(self, sender, receiver, amount, transaction_id = str(uuid.uuid4())):
        # Generate random and unique id.
        transaction = Transaction(sender, receiver, amount, transaction_id)
        transaction.output_index = len(self.transactions)
        self.transactions.append(transaction)
    
    def add_transactions(self, transactions):
        for transaction in transactions:
            self.add_transaction(transaction.sender, transaction.receiver, transaction.amount)
        
    def validate_transactions(self):
        # Check that all transactions are valid
        for transaction in self.transactions:
            if not transaction.is_valid():
                return False
        return True
        
    def clear(self):
        self.transactions = []
        
    def __str__(self):
        text = ''
        for transaction in self.transactions:
            text += str(transaction) + '\n'
            text += '---------------------\n'
        return text
    
class Blockchain:
    def __init__(self):
        self.pending_transactions = TransactionPool() # New attribute
        self.chain = [self.__create_genesis_block().hash]
        self.blocks = {self.chain[0]: self.__create_genesis_block()}
        self.__mining_reward = 10  # For example
        
    # Private method
    def __create_genesis_block(self):
        timestamp = int(time.time())
        self.pending_transactions.add_transaction("0",  "Genesis Wallet", 0)
        hash = calculate_hash(0, "0", timestamp, self.pending_transactions, nonce=0)
        genesis_block = Block(0, "0", timestamp, self.pending_transactions, hash, nonce=0)
        self.pending_transactions.clear()
        return genesis_block

    # Private method
    def __create_new_block(self, previous_block, transactions, nonce=0):
        index = previous_block.index + 1
        timestamp = int(time.time())
        hash = calculate_hash(index, previous_block.hash, timestamp, transactions, nonce)
        return Block(index, previous_block.hash, timestamp, transactions, hash, nonce)
    
    # Private method
    def __add_new_block(self, new_block):
        self.chain.append(new_block.hash)
        self.blocks.update({new_block.hash: new_block})
    
    def get_latest_block(self):
        latest_block_hash = self.chain[-1]
        return self.blocks[latest_block_hash]
    
    def mine_new_block(self, transactions, difficulty, miner_wallet):
        
        # Add the transactions to the list of pending transactions
        self.pending_transactions.add_transactions(transactions)
            
        # Add reward to transactions
        self.reward_miner(miner_wallet)
        
        new_block = self.__create_new_block(self.get_latest_block(), self.pending_transactions)

        # Validate the transactions
        if not self.pending_transactions.validate_transactions():
            print("Invalid transactions, block not created")
            return False
    
        # Mining         
        while new_block.hash[:difficulty] != "0" * difficulty:
            new_block.nonce += 1
            new_block.hash = calculate_hash(new_block.index, new_block.previous_hash, new_block.timestamp, new_block.transactions, new_block.nonce)
        
        self.__add_new_block(new_block)
        print(f'New block #{new_block.index} is mined!')
        
        # Reward the miner        
        reward_transaction = self.pending_transactions.transactions[-1]  # The reward transaction is the last one in the list
        miner_wallet.add_UTXO(reward_transaction)

        # Clear the pending transactions
        self.pending_transactions.clear()
    
    def reward_miner(self, miner_wallet): 
        # Add the reward transaction to the new block
        self.pending_transactions.add_transaction('0', miner_wallet.public_key, self.__mining_reward)
    
    def validate_chain(self):
        for i in range(1, len(self.chain)):
            current_block = self.blocks[self.chain[i]]
            previous_block = self.blocks[self.chain[i - 1]]

            # Check that the hash of the block is correct
            if current_block.hash != calculate_hash(current_block.index, current_block.previous_hash, current_block.timestamp, current_block.transactions, current_block.nonce):
                print('Invalid block')
                return False

            # Check that the previous hash is correct
            if current_block.previous_hash != previous_block.hash:
                print('Invalid chain')
                return False
            
            # Validate transactions
            if not current_block.transactions.validate_transactions():
                print('Invalid transaction')
                return False 
            
        return True

# Miner wallet
# miner_wallet = Wallet()
print("Nouveau portefeuille créé !")
print("Clé privée :", miner_wallet.private_key)
print("Clé publique :", miner_wallet.public_key)
print("Valeur :", miner_wallet.value)   

# Create a new blockchain
blockchain = Blockchain()

# Add new blocks
transactions = [Transaction("Alice", "Bob", 50), Transaction("Bob", "Charlie", 25)]
blockchain.mine_new_block(transactions, 5, miner_wallet)
transactions = [Transaction("Alice", "Eve", 150), Transaction("Alice", "Charlie", 12)]
blockchain.mine_new_block(transactions, 5, miner_wallet)

# Print the blockchain
for hash in blockchain.chain:
    block = blockchain.blocks[hash]
    print(block)

print('Blockchain validation:', blockchain.validate_chain())