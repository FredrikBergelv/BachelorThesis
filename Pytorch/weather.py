import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Simulated Weather Data (Pressure, Temperature, Wind, Rain, Day)
X_train = torch.tensor([
    [1017.81, 5.7, 4, 0, 1],   # Day 1
    [1018.22, 8.5, 5.5, 0, 2],   # Day 2
    [1016.83, 8.9, 4.1, 0, 3],   # Day 3
    [1012.02, 14.0, 2.4, 0, 4],  # Day 4
    [1013.61, 11.9, 3.6, 0, 5],  # Day 5
    [1010.13, 10.2, 5.3, 0, 6],  # Day 6
    [1004.84, 8.8, 6.3, 0, 7]  # Day 7
])

# Next Day's Temperature
y_train = torch.tensor([
    [8.5],  # Temp on Day 3
    [8.9],  # Temp on Day 4
    [14.0], # Temp on Day 5
    [11.9], # Temp on Day 6
    [10.2], # Temp on Day 7
    [8.8],  # Temp on Day 7
    [9.6]   # Temp on Day 8 (prediction target)
])

# Normalize features using Min-Max scaling with epsilon to avoid division by zero
def normalize_data(data):
    min_vals = data.min(dim=0, keepdim=True)[0]
    max_vals = data.max(dim=0, keepdim=True)[0]
    epsilon = 1e-6  # Small value to avoid division by zero
    return (data - min_vals) / (max_vals - min_vals + epsilon)

# Normalize both X_train and y_train
X_train_norm = normalize_data(X_train)
y_train_norm = normalize_data(y_train)

# Define a simple regression model
class WeatherNN(nn.Module):
    def __init__(self):
        super(WeatherNN, self).__init__()
        self.fc1 = nn.Linear(5, 22)  # Input: 5 features → Hidden: 22 neurons
        self.fc2 = nn.Linear(22, 5)  # Hidden: 22 → Hidden: 5
        self.fc3 = nn.Linear(5, 1)   # Hidden: 5 → Output: 1 (temperature)

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
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Reduce learning rate

# Train the model
num_epochs = 10000
for epoch in range(num_epochs):
    model.train()
    
    optimizer.zero_grad()
    predictions = model(X_train_norm)  # Use normalized data for training
    loss = criterion(predictions, y_train_norm)
    
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
    

#%%

# Test the model with a new input
model.eval()

n = 6

new_data = X_train[n]  # Predict for Day 8
new_data_norm = normalize_data(new_data)  # Normalize the new data
predicted_temp_norm = model(new_data_norm).item()

# Denormalize the predicted temperature to get it back to the original scale
predicted_temp = predicted_temp_norm * (y_train.max() - y_train.min()) + y_train.min()

realtemp = y_train[n].item()
print(f'Predicted temperature for next day: {predicted_temp:.2f}°C')
print(f'wherast the real temperature for next day: {realtemp:.2f}°C')
