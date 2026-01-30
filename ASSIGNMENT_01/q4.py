import torch
import torch.nn as nn
import torch.optim as optim

X = torch.tensor([[0,0], [0,1], [1,0], [1,1]], dtype=torch.float32)
Y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

model_xor = nn.Sequential(
    nn.Linear(2, 4),
    nn.Sigmoid(),
    nn.Linear(4, 1),
    nn.Sigmoid()
)

criterion = nn.BCELoss()
optimizer = optim.Adam(model_xor.parameters(), lr=0.1)

for epoch in range(500):
    optimizer.zero_grad()
    outputs = model_xor(X)
    loss = criterion(outputs, Y)
    loss.backward()
    optimizer.step()