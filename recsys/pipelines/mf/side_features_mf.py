import torch
from torch import optim as optim

from recsys.dataset import load_implicit_data
from recsys.models.matrix_factorization import SideFeaturesMF
from recsys.utils import load_model, train_loop


def train(model, data, device, k=10):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 15
    clip_norm = 1.0

    train_loop(
        model,
        data,
        optimizer,
        num_epochs,
        clip_norm,
        device=device,
        k=k,
    )


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)
    data = load_implicit_data(device)
    model = load_model(
        SideFeaturesMF,
        device,
        n_users=data.n_users,
        n_items=data.n_items,
        n_occupations=data.n_occupations,
        embedding_dim=128,
    )
    train(model, data, device, k=10)


if __name__ == "__main__":
    run()
