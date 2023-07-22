import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from recsys.dataset import load_ml1m_data, ImplicitDataOld
from recsys.metrics import ndcg_score, hr_score
from recsys.models.neuralcf import GMF


def train_dataloader(train_inputs, negative_samples, n_negatives, device):
    row_positives, col_positives = train_inputs.nonzero()
    row_negatives = row_positives.repeat(n_negatives)
    col_negatives = np.random.randint(
        0, negative_samples.shape[1], row_negatives.shape[0]
    )

    train_positives = np.vstack(
        [
            row_positives,
            col_positives,
            np.ones((row_positives.shape[0],), dtype=int),
        ]
    ).T
    train_negatives = np.vstack(
        [
            row_negatives,
            negative_samples[row_negatives, col_negatives],
            np.zeros(row_negatives.shape[0]),
        ]
    ).T

    train_data = torch.utils.data.DataLoader(
        np.vstack(
            [
                train_positives,
                train_negatives,
            ]
        ),
        batch_size=1024,
        shuffle=True,
        generator=torch.Generator(device=device),
    )

    return train_data


def train_gmf(data, k=10):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)
    (
        inputs,
        y_true,
        test_codes,
        negative_samples,
    ) = (
        data["inputs"],
        data["labels"],
        data["test_codes"],
        data["negative_samples"],
    )

    test_data = torch.utils.data.DataLoader(
        np.vstack(
            [
                np.arange(y_true.shape[0]).repeat(test_codes.shape[1]),
                test_codes.flatten(),
            ]
        ).T,
        batch_size=2048,
        generator=torch.Generator(device=device),
    )
    model = GMF(*inputs.shape, 128).to(device)

    # Define your model
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 15
    n_negatives = 4

    # Define the gradient clipping value
    max_norm = 1.0

    # Training loop
    train_losses = []
    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_losses = 0

        train_data = train_dataloader(inputs, negative_samples, n_negatives, device)

        for iter_ in tqdm(train_data):
            iter_ = iter_.squeeze(1)
            uids, mids, labels = iter_.int().chunk(3, dim=1)
            optimizer.zero_grad()
            loss = model.loss(
                uids.squeeze(), mids.squeeze(), labels.float(), pointwise=True
            )
            loss.backward()

            # Perform gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()

            train_losses.append(loss.detach().item())
            running_losses += loss.detach().item()

        epoch_loss = running_losses / len(train_data)

        retrieval = []

        model.eval()
        with torch.no_grad():
            for iter_ in tqdm(test_data):
                uids, mids = iter_.chunk(2, dim=1)
                scores = model(uids.squeeze(), mids.squeeze(), pointwise=True)
                retrieval.append(scores.cpu().numpy())

        y_pred = np.take_along_axis(
            test_codes,
            np.argsort(np.vstack(retrieval).reshape(-1, test_codes.shape[1]))[:, ::-1],
            axis=1,
        )[:, :k]

        print(
            f"epoch [{epoch+1}/{num_epochs}], loss: {epoch_loss:.4f}, ndcg: {ndcg_score(y_true, y_pred):.4f}, hr: {hr_score(y_true, y_pred):.4f}"
        )


def run_gmf():
    data = load_ml1m_data()
    train_gmf(data)


if __name__ == "__main__":
    run_gmf()
