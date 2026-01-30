import torch
import torch.nn as nn

class RegressionNN(nn.Module):
    def __init__(self):
        super(RegressionNN, self).__init__()
        self.hidden = nn.Linear(2, 2)
        self.output = nn.Linear(2, 1)

    def forward(self, x):
        x = torch.sigmoid(self.hidden(x))
        x = self.output(x)
        return x

model_reg = RegressionNN()
dummy_input = torch.tensor([[0.5, 0.8]], dtype=torch.float32)
prediction = model_reg(dummy_input)