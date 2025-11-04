# PP-FL: Privacy-Preserving Federated Learning

This repository contains the experimental code, processed data splits, model checkpoints, and logs for the paper:

**"Federated Learning & Privacy-Preserving ML: Decentralized Model Training While Protecting User Data"**  
*(Submitted to IOP Science Journal, 2025)*

---

## 📦 Contents
- `data_prep.py` — generates non-IID client data splits from MNIST and CIFAR-10.
- `train_federated.py` — simulates federated averaging (FedAvg) and privacy-preserving variants.
- `models/` — contains lightweight example model checkpoints.
- `logs/` — includes example training/aggregation logs.
- `requirements.txt` — library dependencies.
- `README.md` — project overview and reproduction steps.

---

## 🧠 Datasets
Publicly available datasets used in this study:
- **MNIST:** http://yann.lecun.com/exdb/mnist/  
- **CIFAR-10:** https://www.cs.toronto.edu/~kriz/cifar.html

No proprietary or sensitive data were used.

---

## 🔁 Reproduction Instructions

```bash
git clone https://github.com/your-username/PP-FL.git
cd PP-FL
pip install -r requirements.txt
python data_prep.py --dataset mnist --num_clients 10 --noniid True
python train_federated.py --config configs/default.yaml

