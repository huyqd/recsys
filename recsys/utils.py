from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from recsys.metrics import compute_metrics


class col:
    user_id = "user_id"
    user_code = "user_code"
    movie_id = "movie_id"
    movie_code = "movie_code"
    rating = "rating"
    timestamp = "timestamp"
    timestamp_rank = "timestamp_rank"
    reverse_timestamp_rank = "reverse_timestamp_rank"
    title = "title"
    genre = "genre"
    gender = "gender"
    age = "age"
    occupation = "occupation"
    zipcode = "zipcode"
    negative = "negative_code"


class path:
    base = Path(__file__).parent.parent
    data = base / "data"

    # ml1m
    ml1m = data / "ml-1m"
    ml1m_ratings = ml1m / "ratings.dat"
    ml1m_users = ml1m / "users.dat"
    ml1m_movies = ml1m / "movies.dat"
    ml1m_npz = ml1m / "ml1m.npz"


def topk(scores, array=None, subset=None, k=10):
    if subset is not None:
        scores = np.take_along_axis(
            scores,
            subset,
            axis=1,
        )
        array = subset

    sorted_scores = np.argsort(scores, axis=1)[:, ::-1][:, :k]

    if array is not None:
        sorted_scores = np.take_along_axis(
            array,
            sorted_scores,
            axis=1,
        )

    return sorted_scores


def load_model(model, device, **kwargs):
    model = model(**kwargs).to(device)

    return model


def eval_loop(model, data, test_dl, k, epoch=None, num_epochs=None, epoch_loss=None):
    model.eval()
    logits = []
    with torch.no_grad():
        for inputs in tqdm(test_dl, desc="test loop", position=2, leave=False):
            scores = model(inputs).cpu().numpy()
            logits.append(scores)
        logits = np.vstack(logits)
        y_pred = topk(logits, array=data.test_codes, k=k)

    if epoch:
        print(f"epoch [{epoch + 1}/{num_epochs}], loss: {epoch_loss:.4f}")
    _ = compute_metrics(data.test_true, y_pred)


def train_loop(model, data, optimizer, num_epochs, clip_norm, device="cpu", k=10):
    train_losses = []

    test_dl = data.create_test_dataloader(batch_size=1024)
    eval_loop(model, data, test_dl, k, epoch=None, num_epochs=None, epoch_loss=None)

    for epoch in tqdm(range(num_epochs), position=0, desc="epoch loop", leave=False):
        model.train()
        running_losses = 0
        train_dl = data.create_negative_sampled_train_dataloader()

        for inputs in tqdm(train_dl, position=1, desc="train loop", leave=False):
            optimizer.zero_grad()
            loss = model.loss(inputs)
            loss.backward()

            # Perform gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)

            optimizer.step()

            train_losses.append(loss.detach().item())
            running_losses += loss.detach().item()

        epoch_loss = running_losses / len(train_dl)

        eval_loop(
            model,
            data,
            test_dl,
            k,
            epoch=epoch,
            num_epochs=num_epochs,
            epoch_loss=epoch_loss,
        )
