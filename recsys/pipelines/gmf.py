import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from recsys.dataset import load_implicit_data, ImplicitData
from recsys.metrics import ndcg_score, hr_score
from recsys.models.nn import GMF


def train_gmf(data, k=10):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)
    (
        inputs,
        labels,
        test_codes,
        negative_samples,
    ) = (
        data["inputs"],
        data["labels"],
        data["test_codes"],
        data["negative_samples"],
    )

    implicit_data = ImplicitData(
        train_inputs=inputs,
        test_inputs=np.hstack(
            [np.arange(labels.shape[0], dtype=int).reshape(-1, 1), test_codes],
            dtype=int,
        ),
        train_batch_size=512,
        test_batch_size=1024,
        add_user_codes=True,
        negative_samples=negative_samples,
        n_negatives=8,
    )

    model = GMF(*inputs.shape, 128).to(device)

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
        train_dl = implicit_data.train_dataloader

        for iter_ in tqdm(train_dl, position=1, desc="train loop", leave=False):
            iter_ = iter_.squeeze(1)
            ucodes, mcodes = iter_[:, 0], iter_[:, 1:]
            train_labels = torch.zeros_like(mcodes, device=device, dtype=float)
            train_labels[:, 0] = 1
            optimizer.zero_grad()
            loss = model.loss(ucodes, mcodes, train_labels)
            loss.backward()

            # Perform gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()

            train_losses.append(loss.detach().item())
            running_losses += loss.detach().item()

        epoch_loss = running_losses / len(train_dl)

        # Print the loss for each epoch
        # if epoch % 10 == 0 or epoch == num_epochs - 1:
        retrieval = []
        model.eval()
        with torch.no_grad():
            for iter_ in tqdm(
                implicit_data.test_dataloader, desc="test loop", position=2, leave=False
            ):
                iter_ = iter_.squeeze(1)
                ucodes, mcodes = iter_[:, 0], iter_[:, 1:]
                scores = model(ucodes, mcodes)
                retrieval.append(scores.cpu().numpy())

            y_pred = np.take_along_axis(
                test_codes,
                np.argsort(np.vstack(retrieval), axis=1)[:, ::-1],
                axis=1,
            )[:, :k]

        print(
            f"epoch [{epoch + 1}/{num_epochs}], loss: {epoch_loss:.4f}, ndcg: {ndcg_score(labels, y_pred):.4f}, hr: {hr_score(labels, y_pred):.4f}"
        )


def run_gmf():
    data = load_implicit_data()
    train_gmf(data)


if __name__ == "__main__":
    run_gmf()
