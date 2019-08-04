import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class DenseLayer(nn.Module):
    def __init__(self, n, growth_rate):
        super(DenseLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=growth_rate*n, out_channels=growth_rate*4, 
                               kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=growth_rate*4, out_channels=growth_rate, 
                               kernel_size=3, stride=1, padding=1)
        
        self.bn1 = nn.BatchNorm2d(growth_rate*n)
        self.bn2 = nn.BatchNorm2d(growth_rate*4)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)
        
        return x
        
class DenseBlock(nn.Module):
    def __init__(self, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList([DenseLayer(n+1, growth_rate) for n in range(num_layers)])
        
    def forward(self, x):        
        for layer in self.layers:
            x = torch.cat((x, layer(x)), 1)
        return x
    
class TransitionBlock(nn.Module):
    def __init__(self, growth_rate, theta, n):
        super(TransitionBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=growth_rate*(n+1), out_channels=growth_rate, 
                               kernel_size=1, stride=1, padding=0)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=math.ceil(1/theta))
        
        self.bn = nn.BatchNorm2d(growth_rate*(n+1))
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.avg_pool(x)
        
        return x
    
class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, theta=0.5, num_layers=[12, 12, 12], num_classes=10):
        super(DenseNet, self).__init__()       
        self.conv = nn.Conv2d(in_channels=3, out_channels=growth_rate, 
                               kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(3)
        self.relu = nn.ReLU(inplace=True)
        
        self.block1 = DenseBlock(growth_rate, num_layers[0])
        self.transitionBlock1 = TransitionBlock(growth_rate, theta, num_layers[0])
        self.block2 = DenseBlock(growth_rate, num_layers[1])
        self.transitionBlock2 = TransitionBlock(growth_rate, theta, num_layers[1])
        self.block3 = DenseBlock(growth_rate, num_layers[2])

        self.avg_pool = nn.AvgPool2d(kernel_size=8, stride=1)
        self.fc = nn.Linear(growth_rate * (num_layers[-1] + 1), num_classes)
                                     
    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        
        x = self.block1(x)
        x = self.transitionBlock1(x)
        
        x = self.block2(x)
        x = self.transitionBlock2(x)
        
        x = self.block3(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x