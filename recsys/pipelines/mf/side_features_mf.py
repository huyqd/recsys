import torch
from torch import optim as optim
from tqdm import tqdm

from recsys.dataset import (
    load_ml1m_data,
    ImplicitData,
)
from recsys.metrics import compute_metrics
from recsys.models.matrix_factorization import SideFeaturesMF
from recsys.utils import topk


def load_data():
    data = load_ml1m_data()
    implicit_data = ImplicitData(data)

    return implicit_data


def load_model(n_users, n_items, n_occupations, embedding_dim, device):
    model = SideFeaturesMF(n_users, n_items, n_occupations, embedding_dim).to(device)

    return model


def train(data, model, device, k=10):
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
        train_dl = data.create_negative_sampled_train_dataloader()

        for inputs in tqdm(train_dl, position=1, desc="train loop", leave=False):
            optimizer.zero_grad()
            loss = model.loss(inputs)
            loss.backward()

            # Perform gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()

            train_losses.append(loss.detach().item())
            running_losses += loss.detach().item()

        epoch_loss = running_losses / len(train_dl)

        model.eval()
        with torch.no_grad():
            scores = (
                model(
                    torch.arange(data.n_users),
                    None,
                    torch.from_numpy(data.user_infos[:, -1]),
                )
                .cpu()
                .numpy()
            )
            y_pred = topk(scores, subset=data.test_codes, k=k)

        print(f"epoch [{epoch + 1}/{num_epochs}], loss: {epoch_loss:.4f}")
        _ = compute_metrics(data.test_labels, y_pred)


def run():
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    data = load_data()
    model = load_model(data.n_users, data.n_items, data.n_occupations, 128, device)
    train(data, model, device, k=10)


if __name__ == "__main__":
    run()
