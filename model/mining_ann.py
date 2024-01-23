# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 10:26:49 2024

@author: ManuMan
"""
import os
import torch
from torch import nn
from torchvision import datasets, transforms
import hashlib

class MiningChain:
    def __init__(self):
        self.__hash_models = []
        self.__models = {}
        self.__filepath = lambda timestamp: f"./data/{timestamp}_mining_classifier.pth"
        os.makedirs("./data/", exist_ok = True)
        
    def get_hash(self, idx):
        return self.__hash_models[idx]
    
    def get_model(self, hash, else_return=None):
        return self.__models.get(hash, else_return)
    
    def get_filepath(self, timestamp):
        return self.__filepath(timestamp)
    
    def synchronize_block(self, new_block, classifier=None):
        if classifier is not None:
            # Save model and update the mining chain
            filepath = self.get_filepath(new_block.timestamp)
            classifier.save_model(filepath)
            self.update_chain(classifier, new_block.timestamp) # Update the mining chain 
            
            # Update block of the blockchain
            hash_model = classifier.get_model_hash() # Hash of parameters
            hash_file = classifier.get_file_hash(filepath) # Hash of the file used to compute mining
            new_block.set_hash_mined(hash_model, hash_file)
        elif classifier is None and len(self.__hash_models)==0:
            self.__hash_models.append(new_block.hash)
            self.__models.update({new_block.hash:{
                                    'index':len(self.__hash_models),
                                    'timestamp':new_block.timestamp,
                                    'hash_file':'0',
                                    'model_location':'None',}
                                   })
        return new_block
        
    def update_chain(self, classifier, timestamp):
        classifier.save_model(self.__filepath(timestamp))
        hash_model = classifier.get_model_hash() # Hash of parameters
        hash_file = classifier.get_file_hash(self.__filepath(timestamp))
        self.__hash_models.append(hash_model)
        self.__models.update({hash_model:{
                                'index':len(self.__hash_models),
                                'timestamp':timestamp,
                                'hash_file':hash_file,
                                'model_location':self.__filepath(timestamp),
                                }
                            })
    
    def validate_model(self, idx):
        
        # Load the model
        original_model_hash = self.get_hash(idx)
        model_info = self.get_model(original_model_hash, None) # Get info associated to the model
        if model_info is None: # If the hash is not present
            return None
        model = self.load_model(model_info)
        
        # Recalculer le hachage du modèle chargé
        model_parameters_bytes = torch.save(model.state_dict(), './data/temp.pth')
        with open('./data/temp.pth', 'rb') as f:
            bytes = f.read() # read entire file as bytes
            loaded_model_hash = hashlib.sha256(bytes).hexdigest()
        
        # Comparer le hachage chargé au hachage original
        if loaded_model_hash == original_model_hash:
            print("The loaded model is the same as the original model.")
            mining_ann = MiningANN()
            mining_ann.model = model
            return mining_ann
        else:
            print("The loaded model is not the same as the original model.")
            return None
    
    def load_model(self, model_info):
        # Charger le dictionnaire à partir du fichier
        checkpoint = torch.load(model_info['model_location'])
        
        # Créer une nouvelle instance du modèle
        model = SimpleNet()

        # Charger les paramètres du modèle à partir du dictionnaire
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 512) # Première couche entièrement connectée
        self.fc2 = nn.Linear(512, 10) # Deuxième couche entièrement connectée

    def forward(self, x):
        x = x.view(-1, 28*28) # Aplatir l'entrée
        x = torch.relu(self.fc1(x)) # Activation ReLU après la première couche
        x = self.fc2(x) # Pas d'activation après la dernière couche
        return x

class MiningANN:
    def __init__(self):
        self.model = SimpleNet()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.epochs = 5
        self.accuracy = 0

    def train(self):
        print('--- A new block is being mined ---')
        trainloader = load_dataset('train')
        for e in range(self.epochs):
            running_loss = 0
            for images, labels in trainloader:
                self.optimizer.zero_grad()

                output = self.model(images)
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
            else:
                print(f"Training loss: {running_loss/len(trainloader)}")
        
    def test(self):
        testloader = load_dataset('test')
        total = 0
        correct = 0
        with torch.no_grad():
            for images, labels in testloader:
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        self.accuracy = correct / total
        print('Accuracy on the test set: %d %%' % (100 * self.accuracy))
        return self.accuracy
        
    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'accuracy': self.accuracy
        }, path)
        
    
    def get_model_hash(self):
        model_parameters = self.model.state_dict()
        model_parameters_bytes = torch.save(model_parameters, './data/temp.pth')
        with open('./data/temp.pth', 'rb') as f:
            bytes = f.read() # read entire file as bytes
            readable_hash = hashlib.sha256(bytes).hexdigest()
        return readable_hash

    def get_file_hash(self, path):
        with open(path, 'rb') as f:
            bytes = f.read() # read entire file as bytes
            readable_hash = hashlib.sha256(bytes).hexdigest()
        return readable_hash

def load_dataset(data_type):
    # Dummy example with MNIST
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    
    # Pour les données d'entraînement
    if data_type == 'train':
        trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
        return trainloader

    # Pour les données de test
    elif data_type == 'test':
        testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
        return testloader
    
    else:
        raise Exception('data_type `{data_type}` not supported, either `train` or `test`.')

# classifier = MiningANN()
# classifier.train()
# classifier.test()

# timestamp = 0
# classifier.save_model(f'./data/{timestamp}_mnist_classifier')
# print('Model hash:', classifier.get_model_hash())
# print('File hash:', classifier.get_file_hash(f'./data/{timestamp}_mnist_classifier'))

# mchain = MiningChain()

# mchain.update_chain(classifier, timestamp)

# hash = mchain.get_hash(0)
# print('model hash:', hash)
# model_info = mchain.get_model(hash, None)
# print(model_info)
# model = mchain.load_model(model_info)
# mining_model = mchain.validate_model(0)
# mining_model.test()