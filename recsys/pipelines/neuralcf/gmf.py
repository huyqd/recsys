import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from recsys.dataset import load_implicit_data, train_dataloader
from recsys.metrics import compute_metrics
from recsys.models.neuralcf import GMF
from recsys.utils import topk


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

    test_dataloader = torch.utils.data.DataLoader(
        np.hstack(
            [np.arange(labels.shape[0], dtype=int).reshape(-1, 1), test_codes],
            dtype=int,
        ),
        batch_size=1024,
        generator=torch.Generator(device=device),
    )

    model = GMF(*inputs.shape, 128).to(device)

    # Define your model
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 15
    n_negatives = 8

    # Define the gradient clipping value
    max_norm = 1.0

    # Training loop
    train_losses = []
    for epoch in tqdm(range(num_epochs), position=0, desc="epoch loop", leave=False):
        model.train()
        running_losses = 0
        train_data = train_dataloader(
            inputs,
            batch_size=512,
            add_user_codes=True,
            negative_samples=negative_samples,
            n_negatives=n_negatives,
            device=device,
        )

        for iter_ in tqdm(train_data, position=1, desc="train loop", leave=False):
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

        epoch_loss = running_losses / len(train_data)

        scores = []
        model.eval()
        with torch.no_grad():
            for iter_ in tqdm(
                test_dataloader, desc="test loop", position=2, leave=False
            ):
                iter_ = iter_.squeeze(1)
                ucodes, mcodes = iter_[:, 0], iter_[:, 1:]
                scores.append(model(ucodes, mcodes).cpu().numpy())

        scores = np.vstack(scores)
        y_pred = topk(scores, array=test_codes, k=k)

        print(f"epoch [{epoch + 1}/{num_epochs}], loss: {epoch_loss:.4f}")
        _ = compute_metrics(labels, y_pred)


def run_gmf():
    data = load_implicit_data()
    train_gmf(data)


if __name__ == "__main__":
    run_gmf()
