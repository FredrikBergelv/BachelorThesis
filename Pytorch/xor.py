#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 10:01:19 2025

@author: fredrik
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 4)  
        self.fc2 = nn.Linear(4, 1)  

    def forward(self, x):
        x = torch.relu(self.fc1(x))  
        x = self.fc2(x)               
        return x
    
X_train = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]) 
y_train = torch.tensor([[0.0], [1.0], [1.0], [0.0]])  

# Instantiate the Model, Define Loss Function and Optimizer
model = SimpleNN()  
criterion = nn.MSELoss()  
optimizer = optim.Adam(model.parameters(), lr=0.01)  

for epoch in range(400):  
    model.train() 

    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)  
    
    # Backward pass and optimize
    optimizer.zero_grad()  
    loss.backward()  
    optimizer.step()  

    if (epoch + 1) % 10 == 0:  
        print(f'Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}')
        
#%%
model.eval()  
with torch.no_grad(): 
    test_data = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0, 0]])
    predictions = np.round(model(test_data))
    print(f'Predictions:\n{predictions}')
    
    
    
