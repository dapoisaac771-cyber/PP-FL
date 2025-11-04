"""
train_federated.py — Simple federated averaging simulation (MNIST small demo).
"""

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

NUM_CLIENTS = 5
LOCAL_BATCH = 32
LOCAL_EPOCHS = 1
LR = 0.01

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.fc(x)

def local_train(state_dict, data_loader):
    model = SimpleNet()
    model.load_state_dict(state_dict)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(LOCAL_EPOCHS):
        for xb, yb in data_loader:
            xb = xb.float()/255.0
            optimizer.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()
    return model.state_dict()

def aggregate(states):
    agg = {}
    for k in states[0].keys():
        agg[k] = sum([s[k] for s in states]) / len(states)
    return agg

if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    clients_states = []
    base_model = SimpleNet()
    base_state = base_model.state_dict()

    for i in range(NUM_CLIENTS):
        # sample small subset per client for demo
        idx = np.random.choice(len(trainset), 1000, replace=False)
        xs = trainset.data[idx].unsqueeze(1).float()
        ys = trainset.targets[idx]
        loader = DataLoader(TensorDataset(xs, ys), batch_size=LOCAL_BATCH, shuffle=True)
        s = local_train(base_state, loader)
        clients_states.append(s)

    new_state = aggregate(clients_states)
    base_model.load_state_dict(new_state)
    print("Federated averaging demo complete.")
