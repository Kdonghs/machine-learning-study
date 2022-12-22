import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# CPU OR GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

#model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),  #Activation function
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.flatten(x)     #nd matrix to 1d matrix
        logits = self.linear_relu_stack(x)
        return logits

#function declaration
model = NeuralNetwork().to(device)

# random test case
X = torch.rand(1, 28, 28, device=device)
logits = model(X)

# softmax:float number to probability
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")