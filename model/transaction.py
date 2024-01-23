# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 10:57:47 2024

@author: ManuMan
"""

import uuid

class Transaction:
    # This is a simple example, a real transaction would be more complex
    def __init__(self, sender, receiver, amount, transaction_id=str(uuid.uuid4())):
        self.id = transaction_id
        self.sender = sender
        self.receiver = receiver
        self.amount = amount
    
    def __str__(self):
        return f"{self.sender} -> {self.receiver}: {self.amount}"

    def is_valid(self, wallets):
        # Check that the sender has enough balance for the transaction
        if wallets.get(self.sender, None):
            if wallets[self.sender].value >= self.amount:
                return True
            else:
                return False
        else:
            return False
    
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
  
    def execute_transaction(self, wallets):        
        nb_transaction = len(self.transactions) 
        for n in range(0, nb_transaction-1): # Skip the last transaction (miner reward)
            transaction = self.transactions[n]
            if transaction.is_valid(wallets):
                wallets[transaction.receiver].add_UTXO(transaction)
                wallets[transaction.sender].remove_UTXO(transaction)
            else:
                print(f"Transaction #{n} | id:{transaction.id} is invalid.")
        return wallets  
  
    def validate_transactions(self, wallets):
        # Check that all transactions are valid
        nb_transaction = len(self.transactions) 
        for n in range(0, nb_transaction-1): # Skip the last transaction (miner reward)
            transaction = self.transactions[n]
            if not transaction.is_valid(wallets):
                print(f"Transaction #{n} | id:{transaction.id} is invalid.")
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


