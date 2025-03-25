import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Generate some simple data (y = 2x + 1)
x = torch.unsqueeze(torch.linspace(-5, 5, 100), dim=1)  # Input data
y = 2 * x + 1 + torch.randn(x.size()) * 0.1  # Output data with some noise

# Define a simple neural network with 1 layer
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # 1 input, 1 output

    def forward(self, x):
        return self.linear(x)

# Initialize model, loss function, and optimizer
model = LinearRegressionModel()
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent

# Training loop
epochs = 1000
for epoch in range(epochs):
    model.train()
    
    # Zero gradients from previous step
    optimizer.zero_grad()

    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x)
    
    # Compute the loss
    loss = criterion(y_pred, y)
    
    # Backward pass: Compute gradients
    loss.backward()
    
    # Update parameters
    optimizer.step()

    # Print loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# After training, let's visualize the result
model.eval()
with torch.no_grad():
    predicted = model(x)

# Plot original data and fitted line
plt.scatter(x.numpy(), y.numpy(), label='Original data')
plt.plot(x.numpy(), predicted.numpy(), label='Fitted line', color='red')
plt.legend()
plt.show()

