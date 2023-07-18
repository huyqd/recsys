import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from recsys.dataset import load_implicit_data
from recsys.metrics import ndcg_score, hr_score
from recsys.models.nn import GMF

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)


def train_dataloader(train_inputs, negative_samples, n_negatives):
    row_positives, train_positives = train_inputs.nonzero()
    row_negatives = row_positives.repeat(n_negatives)
    col_negatives = np.random.randint(
        0, negative_samples.shape[1], row_negatives.shape[0]
    )

    train_negatives = negative_samples[row_negatives, col_negatives].reshape(
        -1, n_negatives
    )

    dl = torch.utils.data.DataLoader(
        np.hstack([train_positives.reshape(-1, 1), train_negatives]),
        batch_size=512,
        shuffle=True,
        generator=torch.Generator(device=device),
    )

    return dl


data = load_implicit_data()
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
        [np.arange(labels.shape[0], dtype=int).reshape(-1, 1), test_codes], dtype=int
    ),
    batch_size=1024,
    generator=torch.Generator(device=device),
)


model = GMF(*inputs.shape, 128).to(device)


# Define your model
criterion = nn.BCEWithLogitsLoss()  # Choose your desired loss function
optimizer = optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 15
n_negatives = 8

# Define the gradient clipping value
max_norm = 1.0

# Training loop
train_losses = []
for epoch in tqdm(range(num_epochs), position=0, desc="epoch loop"):
    model.train()
    running_losses = 0
    train_data = train_dataloader(inputs, negative_samples, n_negatives)

    for iter_ in tqdm(train_data, position=1, desc="train loop"):
        iter_ = iter_.squeeze(1)
        ucodes, mcodes = iter_[:, 0], iter_[:, 1:]
        train_labels = torch.zeros_like(mcodes, device=device, dtype=float)
        train_labels[:, 0] = 1
        optimizer.zero_grad()
        outputs = model(ucodes, mcodes)
        loss = criterion(outputs, train_labels)
        loss.backward()

        # Perform gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        optimizer.step()

        train_losses.append(loss.detach().item())
        running_losses += loss.detach().item()

    epoch_loss = running_losses / len(train_data)

    # Print the loss for each epoch
    # if epoch % 10 == 0 or epoch == num_epochs - 1:
    retrieval = []
    model.eval()
    with torch.no_grad():
        for iter_ in test_dataloader:
            iter_ = iter_.squeeze(1)
            ucodes, mcodes = iter_[:, 0], iter_[:, 1:]
            scores = model(ucodes, mcodes)
            retrieval.append(scores.cpu().numpy())

        y_pred = np.take_along_axis(
            test_codes,
            np.argsort(np.vstack(retrieval), axis=1)[:, ::-1],
            axis=1,
        )[:, :10]

    print(
        f"epoch [{epoch + 1}/{num_epochs}], loss: {epoch_loss:.4f}, ndcg: {ndcg_score(labels, y_pred):.4f}, hr: {hr_score(labels, y_pred):.4f}"
    )
