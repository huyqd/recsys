import numpy as np
import torch
import torch.optim as optim
from tqdm.notebook import tqdm

from recsys.dataset import load_implicit_data
from recsys.metrics import ndcg_score, hr_score
from recsys.models.autoencoder import MultiVAE

device = "cuda"
torch.set_default_device(device)
data = load_implicit_data()
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
train_data = torch.utils.data.DataLoader(
    inputs.toarray().astype(float),
    batch_size=32,
    shuffle=True,
    generator=torch.Generator(device=device),
)

test_data = torch.utils.data.DataLoader(
    inputs.toarray().astype(float),
    batch_size=512,
    generator=torch.Generator(device=device),
)
# %%
decoder_dims = [200, 600, inputs.shape[1]]
encoder_dims = decoder_dims[::-1]
encoder_dims[-1] *= 2
model = MultiVAE(encoder_dims, decoder_dims).to(device)

# Define your model
optimizer = optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 50

# Define the gradient clipping value
max_norm = 1.0

# Training loop
train_losses = []
for epoch in tqdm(range(num_epochs)):
    model.train()
    running_losses = 0

    for inputs in tqdm(train_data):
        optimizer.zero_grad()
        loss = model.negative_sampling_loss(inputs.squeeze().float(), n_negatives=8)
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
        for inputs in test_data:
            scores, _, _ = model(inputs.squeeze().float())
            retrieval.append(scores.cpu().numpy())

        retrieval = np.take_along_axis(np.vstack(retrieval), test_codes, axis=1)

        y_pred = np.take_along_axis(
            test_codes,
            np.argsort(retrieval, axis=1)[:, ::-1],
            axis=1,
        )[:, :10]

    print(
        f"epoch [{epoch+1}/{num_epochs}], loss: {epoch_loss:.4f}, ndcg: {ndcg_score(y_true, y_pred):.4f}, hr: {hr_score(y_true, y_pred):.4f}"
    )
