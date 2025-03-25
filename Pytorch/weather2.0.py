#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 18:42:49 2025

@author: fredrik
"""

import torch
import torch.nn as nn
import torch.optim as optim

# Simulated Weather Data (Pressure, Temperature, Wind, Rain, Day)
X_train = torch.tensor([
    [1017.81, 5.7, 4, 0, 1],  
    [1018.22, 8.5, 5.5, 0, 2],  
    [1016.83, 8.9, 4.1, 0, 3],  
    [1012.02, 14.0, 2.4, 0, 4],  
    [1013.61, 11.9, 3.6, 0, 5],  
    [1010.13, 10.2, 5.3, 0, 6],  
    [1004.84, 8.8, 6.3, 0, 7]  
], dtype=torch.float32)  # Ensure float32 type

# Target: Next day's full weather data
y_train = torch.tensor([
    [1018.22, 8.5, 5.5, 0, 2],  
    [1016.83, 8.9, 4.1, 0, 3],  
    [1012.02, 14.0, 2.4, 0, 4],  
    [1013.61, 11.9, 3.6, 0, 5],  
    [1010.13, 10.2, 5.3, 0, 6],  
    [1004.84, 8.8, 6.3, 0, 7],  
    [1008.43, 6.0, 13.7, 0, 8]  # Future day prediction target
], dtype=torch.float32)

# Function to normalize data (using training set statistics)
def normalize(data, mean, std):
    return (data - mean) / (std + 1e-6)  # Add epsilon to avoid division by zero

# Compute mean and std from the training set
X_mean, X_std = X_train.mean(dim=0), X_train.std(dim=0)
y_mean, y_std = y_train.mean(dim=0), y_train.std(dim=0)

# Normalize the dataset
X_train_norm = normalize(X_train, X_mean, X_std)
y_train_norm = normalize(y_train, y_mean, y_std)

print("Normalized X_train:", X_train_norm)

# Define a simple regression model for predicting full weather data
class WeatherNN(nn.Module):
    def __init__(self):
        super(WeatherNN, self).__init__()
        self.fc1 = nn.Linear(5, 10)  # Input: 5 features → Hidden: 10 neurons
        self.fc2 = nn.Linear(10, 5)  # Hidden: 10 → Hidden: 5
        self.fc3 = nn.Linear(5, 5)   # Output: 5 values (Pressure, Temp, Wind, Rain, Day)

        # Initialize weights with Xavier initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # First hidden layer
        x = torch.relu(self.fc2(x))  # Second hidden layer
        x = self.fc3(x)  # Output layer
        return x

# Initialize model, loss function, and optimizer
model = WeatherNN()
criterion = nn.MSELoss()  # Mean Squared Error for regression
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Reduce learning rate

# Train the model
num_epochs = 10000
for epoch in range(num_epochs):
    model.train()
    
    optimizer.zero_grad()
    predictions = model(X_train_norm)  # Use normalized data for training
    loss = criterion(predictions, y_train_norm)  # Now predicting all 5 outputs
    
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
    
    if torch.isnan(loss):  # Stop if loss goes to NaN
        print("Training stopped due to NaN loss!")
        break

#%%

# Test the model with a new input
model.eval()

n = 3   # Predict for the 7th data point

new_data = X_train[n].unsqueeze(0)  # Add batch dimension
new_data_norm = normalize(new_data, X_mean, X_std)  # Normalize using training stats
predicted_next_day_norm = model(new_data_norm).squeeze(0)  # Remove batch dimension

# Denormalize the predicted values
predicted_next_day = predicted_next_day_norm * y_std + y_mean

real_next_day = y_train[n]  # Actual next day values
print("\nPredicted Weather for Next Day:")
print(f"Pressure: {predicted_next_day[0]:.2f} hPa")
print(f"Temperature: {predicted_next_day[1]:.2f}°C")
print(f"Wind Speed: {predicted_next_day[2]:.2f} m/s")
print(f"Rain: {predicted_next_day[3]:.2f} mm")
print(f"Day: {predicted_next_day[4]:.0f}")

print("\nActual Weather for Next Day:")
print(f"Pressure: {real_next_day[0]:.2f} hPa")
print(f"Temperature: {real_next_day[1]:.2f}°C")
print(f"Wind Speed: {real_next_day[2]:.2f} m/s")
print(f"Rain: {real_next_day[3]:.2f} mm")
print(f"Day: {real_next_day[4]:.0f}")
