"""
data_prep.py — Create IID and non-IID data partitions for federated learning.
"""

import numpy as np
import os
from torchvision import datasets, transforms

def create_partitions_mnist(num_clients=10, iid=False, out_dir="client_data"):
    transform = transforms.Compose([transforms.ToTensor()])
    root = "./data"
    dataset = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    data = dataset.data.numpy()
    targets = dataset.targets.numpy()

    if iid:
        indices = np.arange(len(data))
        np.random.shuffle(indices)
        split_indices = np.array_split(indices, num_clients)
    else:
        # Non-IID: group by label then split
        split_indices = []
        labels = np.arange(10)
        label_indices = [np.where(targets == l)[0] for l in labels]
        # simple round-robin assignment across clients from each label
        client_indices = [[] for _ in range(num_clients)]
        for l_idx in label_indices:
            np.random.shuffle(l_idx)
            parts = np.array_split(l_idx, num_clients)
            for c in range(num_clients):
                client_indices[c].extend(parts[c].tolist())
        split_indices = [np.array(ci) for ci in client_indices]

    os.makedirs(out_dir, exist_ok=True)
    for i, idx in enumerate(split_indices):
        np.savez(os.path.join(out_dir, f"client_{i}.npz"), x=data[idx], y=targets[idx])
    print(f"Created {num_clients} {'IID' if iid else 'non-IID'} partitions in {out_dir}/")

if __name__ == "__main__":
    create_partitions_mnist(num_clients=10, iid=False)
