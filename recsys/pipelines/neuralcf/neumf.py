import torch
from torch import optim as optim

from recsys.dataset import load_implicit_data
from recsys.models.neuralcf import NeuMF
from recsys.utils import load_model, train_loop


def train(model, data, k=10):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 15
    clip_norm = 1.0

    train_loop(model, data, optimizer, num_epochs, clip_norm, k=k)


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    torch.set_default_device(device)
    data = load_implicit_data(device, train_batch_size=512, test_batch_size=1024)
    model = load_model(
        NeuMF,
        device,
        n_users=data.n_users,
        n_items=data.n_items,
        embedding_dim=128,
    )
    train(model, data, k=10)


if __name__ == "__main__":
    run()
