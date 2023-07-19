import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from recsys.dataset import load_implicit_data
from recsys.metrics import ndcg_score, hr_score
from recsys.pipelines.mf.vanilla_mf import run_vanillamf
from recsys.models.matrix_factorization import BiasMF


def train_biasmf(data, k=10):
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

    train_data = torch.utils.data.DataLoader(
        np.hstack(
            [
                np.arange(inputs.shape[0]).reshape(-1, 1),
                inputs.toarray(),
            ]
        ),
        batch_size=32,
        shuffle=True,
        generator=torch.Generator(device=device),
    )
    model = BiasMF(*inputs.shape, 128).to(device)

    # Define your model
    optimizer = optim.SGD(model.parameters(), lr=100)
    num_epochs = 500

    # Define the gradient clipping value
    max_norm = 1.0

    # Training loop
    train_losses = []
    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_losses = 0

        for inter_ in train_data:
            inter_ = inter_.squeeze(1)
            uids, labels = inter_[:, 0], inter_[:, 1:]
            optimizer.zero_grad()
            loss = model.loss(uids.int(), labels.float())
            loss.backward()

            # Perform gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()

            train_losses.append(loss.detach().item())
            running_losses += loss.detach().item()

        epoch_loss = running_losses / len(train_data)

        # print the loss for each epoch
        if epoch % 20 == 0 or epoch == num_epochs - 1:
            model.eval()
            with torch.no_grad():
                retrieval = (
                    F.softmax(model(torch.arange(inputs.shape[0])), dim=1).cpu().numpy()
                )
                y_pred = np.take_along_axis(
                    test_codes,
                    np.argsort(
                        np.take_along_axis(retrieval, test_codes, axis=1), axis=1
                    )[:, ::-1],
                    axis=1,
                )[:, :k]

            print(
                f"epoch [{epoch+1}/{num_epochs}], loss: {epoch_loss:.4f}, ndcg@{k}: {ndcg_score(y_true, y_pred):.4f}, hr@{k}: {hr_score(y_true, y_pred):.4f}"
            )


def run_biasmf():
    data = load_implicit_data()
    train_biasmf(data)


if __name__ == "__main__":
    run_biasmf()
