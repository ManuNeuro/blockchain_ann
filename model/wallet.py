# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 11:00:16 2024

@author: ManuMan
"""

import sys
from pathlib import Path

# The model and key generator
from blockchain_ann.model.public_key_generator import generate_public_key
from blockchain_ann.model.private_key_generator import generate_private_key, fake_private_key
from blockchain_ann.model.transaction import Transaction

class WalletPool():
    def __init__(self):
        self.wallets = {}
        self.index = {}
        
    def create_wallet(self, fake, value):
        wallet = Wallet(fake) # Create the fake wallet
        wallet.add_UTXO(Transaction("0", wallet.public_key, value)) # Add a UTXO for testing
        self.wallets.update({wallet.public_key:wallet})
        idx = len(self.index)
        self.index.update({idx:wallet.public_key})

    def create_fake_wallets(self, n, value):
        for i in range(n):
            self.create_wallet(fake=True, value=value)
    
    def by_index(self, idx):
        return self.wallets.get(self.index[idx], None)
    
    def __str__(self):
        text = '--- Wallets --- \n'
        for wallet in self.wallets:
            text += str(wallet) + '\n'
            text+= '---------------- \n'
        return text
    
class Wallet:
    def __init__(self, fake=False):
        self.private_key, self.public_key = self.generate_keys(fake)
        self.UTXOs = {}  # UTXOs owned by this wallet
        self.value = 0.0  # Value of UTXOs owned by this wallet

    def generate_keys(self, fake=False):
        # Generate a new private/public key pair
        if not fake:
            private_key = generate_private_key(1024, save=False)
        else:
            private_key = fake_private_key(1024)
        public_key = generate_public_key(private_key['private_key(int)'], save=False)
        return (private_key['private_key(hex)'],
                public_key['public_key(hex)'])

    def update_value(self):
        # Update the value of UTXOs owned by this wallet
        self.value = sum(UTXO.amount for UTXO in self.UTXOs.values())

    def add_UTXO(self, transaction):
        # Create a new UTXO from the transaction
        utxo = UTXO(transaction.id, len(self.UTXOs)+1, self.public_key, transaction.amount)

        # Add the UTXO to the wallet
        self.UTXOs.update({utxo.transaction_id: utxo})

        # Update the value of the wallet
        self.update_value()
    
    def remove_UTXO(self, transaction):
        # Create a new UTXO from the transaction
        utxo = UTXO(transaction.id, len(self.UTXOs)+1, self.public_key, -1*transaction.amount)

        # Add the UTXO to the wallet
        self.UTXOs.update({utxo.transaction_id: utxo})

        # Update the value of the wallet
        self.update_value()
    
    def __str__(self):
        text = f"Public key: {self.public_key} \n"
        text += 'Value in wallet: {self.value} \n'
        return text
    
class UTXO:
    def __init__(self, transaction_id, output_index, owner, amount):
        self.transaction_id = transaction_id
        self.output_index = output_index
        self.owner = owner
        self.amount = amount
    
    def __str__(self):
        text = f"id:{self.transaction_id} | Public key: {self.owner} \n"
        text += f"UTXO value: {self.amount} \n"
        return text
    