# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 11:20:19 2024

@author: ManuMan
"""
import sys
from pathlib import Path
PATH_ROOT = Path(__file__).parents[1]  # Adjust the number if needed
sys.path.append(str(PATH_ROOT))

from blockchain_ann.model.transaction import Transaction
from blockchain_ann.model.mining_ann import MiningANN, MiningChain
from blockchain_ann.model.blockchain import Blockchain, validate_chain
from blockchain_ann.model.wallet import Wallet, WalletPool

# Generate fake wallets
wallets = WalletPool()
wallets.create_fake_wallets(10, value=10)
print(wallets)

# Generate fake transactions
transactions = [Transaction(wallets.by_index(0).public_key,  
                            wallets.by_index(2).public_key, 3), 
                Transaction(wallets.by_index(2).public_key,  
                            wallets.by_index(3).public_key, 2),
                Transaction(wallets.by_index(4).public_key,  
                            wallets.by_index(3).public_key, 1),
                Transaction(wallets.by_index(5).public_key,  
                            wallets.by_index(4).public_key, 10)] 

# Create the mining chain
mining_chain = MiningChain()

## Create the blockchain
blockchain = Blockchain(mining_chain)

# Add new blocks
miner_public_key = wallets.by_index(9).public_key
blockchain.mine_new_block(transactions, 90, mining_chain, wallets.wallets, miner_public_key)

print(' ### Blockchain ###')
for block_hash in blockchain.chain:
    block = blockchain.blocks[block_hash]
    print(block)

print(' ### Mining side chain ###')
for idx in range(blockchain.block_height+1):
    hash_model = mining_chain.get_hash(idx)
    model_info = mining_chain.get_model(hash_model)
    print(hash_model, model_info)

print('----------------------')
print('Blockchain validation:', validate_chain(blockchain, mining_chain, wallets.wallets))