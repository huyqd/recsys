import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.notebook import tqdm

from recsys.dataset import load_ml1m_data
from recsys.metrics import compute_metrics
from recsys.models.autoencoder import CDAE
from recsys.utils import topk

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)
data = load_ml1m_data()
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
array = np.hstack(
    [
        np.arange(inputs.shape[0]).reshape(-1, 1),
        inputs.toarray(),
    ]
)
train_data = torch.utils.data.DataLoader(
    array,
    batch_size=256,
    shuffle=True,
    generator=torch.Generator(device=device),
)

test_data = torch.utils.data.DataLoader(
    array,
    batch_size=512,
    generator=torch.Generator(device=device),
)
model = CDAE(*inputs.shape, 512, 0.2).to(device)

# Define your model
criterion = nn.BCEWithLogitsLoss()  # Choose your desired loss function
optimizer = optim.Adam(model.parameters(), lr=1e-2)
num_epochs = 50

# Define the gradient clipping value
max_norm = 1.0

# Training loop
train_losses = []
for epoch in tqdm(range(num_epochs), position=0, desc="epoch loop", leave=False):
    model.train()
    running_losses = 0

    for iter_ in tqdm(train_data, position=1, desc="train loop", leave=False):
        uids, mids = iter_[:, 0], iter_[:, 1:]
        optimizer.zero_grad()
        outputs = model(uids.int(), mids.float())
        loss = criterion(outputs, mids.float())
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
        for iter_ in test_data:
            iter_ = iter_.squeeze(1)
            uids, mids = iter_[:, 0], iter_[:, 1:]
            scores.append(model(uids.int(), mids.float()).cpu().numpy())

    scores = np.vstack(scores)
    y_pred = topk(scores, subset=test_codes, k=10)

    print(f"epoch [{epoch + 1}/{num_epochs}], loss: {epoch_loss:.4f}")
    _ = compute_metrics(y_true, y_pred)
