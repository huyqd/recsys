from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

CURRENT_PATH = Path(__file__).cwd()


class BinaryML1mDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=128, n_negative_samples=4, n_workers=8):
        super().__init__()
        self.batch_size = batch_size
        self.n_negative_samples = n_negative_samples
        self.n_workers = n_workers
        self.data_dir = CURRENT_PATH / "data" / "ml-1m" / "binary"

        self.n_users, self.n_items = torch.load(self.data_dir / 'ml-1m-train.pt').size()
        self.train_sparse = None
        self.train_score = None
        self.test_users = None
        self.test_items = None
        self.test_labels = None

        self.train_ds = None
        self.test_ds = None

        self.train_steps = None
        self.test_steps = None

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            # Train data
            self.train_sparse = torch.load(self.data_dir / 'ml-1m-train.pt')
            self.train_score = torch.sparse.sum(self.train_sparse, dim=0)

            # Test data
            test_pos = torch.load(self.data_dir / 'ml-1m-test-pos.pt')
            test_neg = torch.load(self.data_dir / 'ml-1m-test-neg.pt')

            test_items = []
            for u in range(self.n_users):
                items = torch.cat((test_pos[u, 1].view(1), test_neg[u]))
                test_items.append(items)

            self.test_items = torch.vstack(test_items)
            self.test_labels = torch.zeros(self.test_items.shape)
            self.test_labels[:, 0] += 1
            self.test_users = test_pos[:, 0].view(-1, 1).repeat(1, self.test_items.shape[1])

        if stage == "test" or stage is None:
            pass

    def _negative_sampling(self):
        pos = self.train_sparse._indices().T
        user_ids = torch.unique(pos[:, 0])

        users = []
        items = []
        labels = []
        for user in user_ids:
            pos_items = pos[pos[:, 0] == user, 1]
            n_pos_items = pos_items.shape[0]

            # Sample negative items
            n_neg_items = self.n_negative_samples * n_pos_items
            sampling_prob = torch.ones(self.n_items)
            sampling_prob[pos_items] = 0  # Don't sample positive items
            neg_items = torch.multinomial(sampling_prob, n_neg_items, replacement=True)

            users.append(user.repeat(n_pos_items + n_neg_items))
            items.append(torch.cat([pos_items, neg_items]))
            labels.append(torch.cat([torch.ones(n_pos_items), torch.zeros(n_neg_items)]))

        users = torch.cat(users)
        items = torch.cat(items)
        labels = torch.cat(labels)

        return users, items, labels

    def train_dataloader(self):
        users, items, labels = self._negative_sampling()
        self.train_ds = BinaryML1mDataset(users, items, labels)

        train_dl = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.n_workers)
        self.train_steps = len(train_dl)

        return train_dl

    def val_dataloader(self):
        self.test_ds = BinaryML1mDataset(self.test_users, self.test_items, self.test_labels)

        test_dl = DataLoader(self.test_ds, batch_size=512, shuffle=False, num_workers=self.n_workers)
        self.test_steps = len(test_dl)

        return test_dl


class RatingML1mDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=128, n_workers=8):
        super().__init__()
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.data_dir = CURRENT_PATH / "data" / "ml-1m"

        self.ratings_sparse = self._get_sparse_ratings()
        self.n_users, self.n_items = self.ratings_sparse.size()

        self.train_data = None
        self.test_data = None

        self.train_ds = None
        self.test_ds = None

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            idx, values = self.ratings_sparse.indices().T, self.ratings_sparse.values()
            train_idx, test_idx, train_values, test_values = train_test_split(idx, values, test_size=0.1)
            self.train_data = torch.sparse_coo_tensor(train_idx.T, train_values).coalesce().to_dense()
            self.test_data = torch.sparse_coo_tensor(test_idx.T, test_values).coalesce().to_dense()

        if stage == "test" or stage is None:
            pass

    def _get_sparse_ratings(self):
        ratings = pd.read_csv(self.data_dir / "ratings.dat",
                              sep="::",
                              header=None,
                              names=['user_id', 'movie_id', 'rating', 'timestamp'],
                              dtype={'user_id': np.int32, 'movie_id': np.int32,
                                     'ratings': np.float32, 'timestamp': np.int64},
                              engine='python',
                              encoding="ISO-8859-1")

        idx = ratings[['user_id', 'movie_id']].sub(1).values.T
        values = ratings['rating'].values
        sparse_ratings = torch.sparse_coo_tensor(idx, values).coalesce()

        return sparse_ratings

    def train_dataloader(self):
        self.train_ds = RatingML1mDataset(self.train_data)

        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.n_workers)

    def val_dataloader(self):
        self.test_ds = RatingML1mDataset(self.test_data)

        return DataLoader(self.test_ds, batch_size=self.batch_size * 2, shuffle=False, num_workers=self.n_workers)


class BinaryML1mDataset(Dataset):
    def __init__(self, users, items, labels=None):
        assert users.shape == items.shape
        if labels is not None:
            assert users.shape == labels.shape
        self.users = users
        self.items = items
        self.labels = labels

    def __len__(self):
        return self.users.shape[0]

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.users[idx], self.items[idx], self.labels[idx]
        else:
            return self.users[idx], self.items[idx], None


class RatingML1mDataset(Dataset):
    def __init__(self, data):
        self.data = data.T.float()

    def __len__(self):
        return self.data.size()[0]

    def __getitem__(self, idx):
        return self.data[idx]
