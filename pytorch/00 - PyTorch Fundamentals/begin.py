import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np 
import matplotlib.pyplot as plt

"""
x_values = [i for i in range(11)]
print(x_values)
print(type(x_values))
x_train = np.array(x_values, dtype=np.float64)
x_train = x_train.reshape(-1,1)
"""


torch.manual_seed(0)
X_train = torch.randn(10,2)
y_train = X_train * 2 +3 + 0.1*torch.randn(10,2)

class LinearRegressinModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LinearRegressinModel, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out 
        

input_dim = 2
hidden_dim = 10
output_dim = 2
model = LinearRegressinModel(input_dim, hidden_dim, output_dim)

print(model)

##
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

##
num_epochs = 1000
losses = []

for epoch in range(num_epochs):
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


with torch.no_grad(): 
    predicted = model(X_train).detach().numpy()

plt.plot(X_train.numpy(), y_train.numpy(), 'ro', label='Original data')
plt.plot(X_train.numpy(), predicted, label='Fitted line')
plt.legend()
plt.show()

print(X_train)
print(predicted)