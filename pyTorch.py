import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import nnfs
from nnfs.datasets import spiral_data

# this is the equivalent code to the one in nnFromScratch.py but using PyTorch

nnfs.init()

X, y = spiral_data(100, 3)
X = torch.tensor(X).to("cuda").float()
y = torch.tensor(y).to("cuda").long()

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 3)
    
    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return X
    
net = Model().to("cuda")
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

for epoch in range(10001):
    output = net(X)
    loss = loss_function(output, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if not epoch % 100:
            print(f'epoch: {epoch}, ' + f'loss: {loss:.3f}')

X_test, y_test = spiral_data(100, 3)
X_test = torch.tensor(X_test).to("cuda").float()
y_test = torch.tensor(y_test).to("cuda").long()

output = net(X_test)
loss = loss_function(output, y_test)
predictions = torch.argmax(output, dim=1)
accuracy = (predictions == y_test).float().mean()
print("Accuracy: ", accuracy.item())

