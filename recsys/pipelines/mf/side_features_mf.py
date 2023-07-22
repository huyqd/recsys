import torch
from torch import optim as optim
from tqdm import tqdm

from recsys.dataset import (
    train_dataloader,
    load_ml1m_data,
    create_negative_sampled_train_data,
)
from recsys.metrics import compute_metrics
from recsys.models.matrix_factorization import SideFeaturesMF
from recsys.utils import topk
import numpy as np


class Ml1mDataset(torch.utils.data.Dataset):
    def __init__(self, data, infos, labels):
        self.data = data
        self.infos = infos
        self.labels = labels
        self.n_users = data.shape[0]
        self.n_items = data.shape[1] - 1
        self.n_occupations = np.unique(self.infos[:, -1]).shape[0]

    def __len__(self):
        return self.n_users

    def __getitem__(self, idx):
        items = {
            "user_code": self.data[idx, 0],
            "movie_code": self.data[idx, 1:],
            "user_occupation": self.infos[idx, -1],
            "label": self.labels[idx],
        }

        return items


def load_data():
    data = load_ml1m_data()
    train_data, train_labels = create_negative_sampled_train_data(
        data["implicit_matrix"], data["negative_samples"], 4
    )

    train_user_infos = data["user_infos"][train_data[:, 0]]
    n_users, n_items = data["implicit_matrix"].shape

    return Ml1mDataset(train_data, train_user_infos, train_labels), n_users, n_items


def load_model(n_users, n_items, n_occupations, embedding_dim, device):
    model = SideFeaturesMF(n_users, n_items, n_occupations, embedding_dim).to(device)

    return model


def train(data, model, device, k=10):
    train_data = torch.utils.data.DataLoader(
        data,
        batch_size=512,
        shuffle=True,
        # generator=torch.Generator(device=device),
    )

    # Define your model
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 15

    # Define the gradient clipping value
    max_norm = 1.0

    # Training loop
    train_losses = []
    for epoch in tqdm(range(num_epochs), position=0, desc="epoch loop", leave=False):
        model.train()
        running_losses = 0

        for inputs in tqdm(train_data, position=1, desc="train loop", leave=False):
            optimizer.zero_grad()
            loss = model.loss(inputs)
            loss.backward()

            # Perform gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()

            train_losses.append(loss.detach().item())
            running_losses += loss.detach().item()

        epoch_loss = running_losses / len(train_data)

        model.eval()
        with torch.no_grad():
            scores = model(torch.arange(inputs.shape[0])).cpu().numpy()
            y_pred = topk(scores, subset=test_codes, k=k)

        print(f"epoch [{epoch + 1}/{num_epochs}], loss: {epoch_loss:.4f}")
        _ = compute_metrics(y_true, y_pred)


def run():
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    data, n_users, n_items = load_data()
    model = load_model(n_users, n_items, data.n_occupations, 128, device)
    train(data, model, device, k=10)


if __name__ == "__main__":
    run()
