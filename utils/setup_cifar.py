
import tensorflow as tf
import copy
import torch.nn.functional as F
from keras.datasets import cifar10
import random
import warnings
import torch.nn as nn
import torchvision
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.optim as optim
from sklearn.manifold import TSNE
from torch import linalg as LA
import os



class CIFAR:
    def __init__(self, device):
        self.device = device
        self.mean = torch.tensor([0.4914, 0.4822, 0.4465], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.2023, 0.1994, 0.2010], device=self.device).view(1, 3, 1, 1)
        self.transform_train = transforms.Compose([ 
            transforms.RandomCrop(32, padding=4),   
            transforms.RandomHorizontalFlip(),      
        ])

        self.to_tensor = transforms.ToTensor() 

        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=False
        )

        data = trainset.data  
        labels = trainset.targets  


        print(f"Data shape: {data.shape}")
        print(f"Labels length: {len(labels)}")


        self.test_data = self._process_data(data)  
        self.test_labels =  torch.tensor(trainset.targets, device=self.device)  


        self.label_name = [
            'plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]


    def _process_data(self, data):

        data_tensor = torch.tensor(data, dtype=torch.float32, device=self.device).permute(0, 3, 1, 2) / 255.0 


        transform_pipeline = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Lambda(lambda x: x - 0.5) 
        ])

        transformed_data = torch.stack([transform_pipeline(img) for img in data_tensor]).to(self.device)

        return transformed_data





############################ model weights：./models/cifar10-resnet18.pth

class ResidualBlock(nn.Module):
    def __init__(self, in_len, out_len, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_len, out_len, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_len)
        self.conv2 = nn.Conv2d(out_len, out_len, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_len)
        
        if stride != 1 or in_len != out_len:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_len, out_len, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_len)
            )
        else:
            self.shortcut = nn.Identity()  
        

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x))) 
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  
        return F.relu(out, inplace=True)  

    
    
class ResNet18_model(nn.Module):
    def __init__(self):
        super(ResNet18_model, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
                
        self.layer1 = nn.Sequential(
            ResidualBlock(64, 64, stride=1),
            ResidualBlock(64, 64, stride=1)
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128, stride=1)
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256, stride=1)
        )
        
        self.layer4 = nn.Sequential(
            ResidualBlock(256, 512, stride=2),
            ResidualBlock(512, 512, stride=1)
        )
        
        self.linear = nn.Linear(512, 10)

    def create_block(self, out_len, stride):
        layers = []
        layers.append(ResidualBlock(self.in_len, out_len, stride=stride))
        self.in_len = out_len
        layers.append(ResidualBlock(self.in_len, out_len, stride=1))
        return nn.Sequential(*layers)
    
    def get_feature_space(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        return out.view(out.size(0), -1)

    
    def forward(self, x):
        out = self.get_feature_space(x)
        return self.linear(out)  


    def predict(self, inputs):
        """Perform predictions on input data while keeping gradient tracking optional."""
        self.eval()
        outputs = self(inputs)
        return F.softmax(outputs, dim=-1)


    def save_model(self, save_dir):
        """Save model parameters to the specified directory."""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model_path = os.path.join(save_dir, "cifar10-resnet18.pth")
        torch.save(self.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    def load_model(self, restore_dir):
        """Load model parameters from the specified directory."""
        model_path = os.path.join(restore_dir, "cifar10-resnet18.pth")
        if os.path.exists(model_path):
            self.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            print(f"Model restored from {model_path}")
        else:
            raise FileNotFoundError(f"No model file found at {model_path}")