#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 18:46:24 2025

@author: fredrik
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 18:42:49 2025

@author: fredrik
"""

import torch
import torch.nn as nn
import torch.optim as optim
from Pytorch import load_csv as csv

# Simulated Weather Data (Pressure, Temperature, Wind, Rain, Day)
X_train = csv.features.to(torch.float32)
y_train = csv.targets.to(torch.float32)

# Compute mean and std
X_mean, X_std = X_train.mean(dim=0).to(torch.float32), X_train.std(dim=0).to(torch.float32)
y_mean, y_std = y_train.mean(dim=0).to(torch.float32), y_train.std(dim=0).to(torch.float32)

# Normalize the dataset
X_train_norm = (X_train - X_mean) / (X_std + 1e-6)
y_train_norm = (y_train - y_mean) / (y_std + 1e-6)

print("Normalized X_train:", X_train_norm)

#%% 
# Define a simple regression model for predicting full weather data
class WeatherNN(nn.Module):
    def __init__(self):
        super(WeatherNN, self).__init__()
        self.fc1 = nn.Linear(4, 10)  # Input: 5 features → Hidden: 10 neurons
        self.fc2 = nn.Linear(10, 4)  # Hidden: 10 → Hidden: 5
        self.fc3 = nn.Linear(4, 4)   # Output: 5 values (Pressure, Temp, Wind, Rain, Day)

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
    

#%%
# Test the model with a new input
model.eval()

n = 980   # Predict for the 7th data point

# During inference, ensure float32 type
new_data = X_train[n].unsqueeze(0).to(torch.float32)
new_data_norm = (new_data - X_mean) / (X_std + 1e-6)
predicted_next_day_norm = model(new_data_norm).squeeze(0)  # Remove batch dimension

# Denormalize the predicted values
predicted_next_day = predicted_next_day_norm * y_std + y_mean

real_next_day = y_train[n]  # Actual next day values
print("\nPredicted Weather for Next Day:")
print(f"Pressure: {predicted_next_day[0]:.0f} hPa")
print(f"Rain: {predicted_next_day[1]:.2} mm")
print(f"Temperature: {predicted_next_day[2]:.0f}°C")
print(f"Wind Speed: {predicted_next_day[3]:.1f} m/s")

print("\nReal Weather for Next Day:")
print(f"Pressure: {real_next_day[0]:.0f} hPa")
print(f"Rain: {real_next_day[1]:.2} mm")
print(f"Temperature: {real_next_day[2]:.0f}°C")
print(f"Wind Speed: {real_next_day[3]:.1f} m/s")


#%%
new_data = torch.tensor([[1011, 0, 13, 4.8]])
new_data_norm = (new_data - X_mean) / (X_std + 1e-6)
predicted_next_day_norm = model(new_data_norm).squeeze(0)  # Remove batch dimension

# Denormalize the predicted values
predicted_next_day = predicted_next_day_norm * y_std + y_mean

real_next_day = y_train[n]  # Actual next day values
print("\nPredicted Weather for Next Day:")
print(f"Pressure: {predicted_next_day[0]:.0f} hPa")
print(f"Rain: {predicted_next_day[1]:.2} mm")
print(f"Temperature: {predicted_next_day[2]:.0f}°C")
print(f"Wind Speed: {predicted_next_day[3]:.1f} m/s")
