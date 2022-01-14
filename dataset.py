from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader

CURRENT_PATH = Path(__file__).cwd()


class ML1mDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=128, n_negative_samples=4, n_workers=8):
        super().__init__()
        self.batch_size = batch_size
        self.n_negative_samples = n_negative_samples
        self.n_workers = n_workers
        self.data_dir = CURRENT_PATH / "data"

        self.n_users = None
        self.n_items = None
        self.train_sparse = None
        self.train_score = None
        self.test_users = None
        self.test_items = None
        self.test_labels = None

        self.train_ds = None
        self.test_ds = None

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            # Train data
            self.train_sparse = torch.load(self.data_dir / 'ml-1m-train.pt')
            self.n_users, self.n_items = self.train_sparse.size()

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
            sampling_prob[pos_items] = 0    # Don't sample positive items
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
        self.train_ds = ML1mDataset(users, items, labels)

        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.n_workers)

    def val_dataloader(self):
        self.test_ds = ML1mDataset(self.test_users, self.test_items, self.test_labels)

        return DataLoader(self.test_ds, batch_size=512, shuffle=False, num_workers=self.n_workers)


class ML1mDataset(Dataset):
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
