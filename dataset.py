from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader

CURRENT_PATH = Path(__file__).cwd()


class ML1mDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=128, n_workers=8):
        super().__init__()
        self.batch_size = batch_size
        self.n_workers = n_workers

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        self.train_ds = TrainDataset()
        self.test_ds = TestDataset()

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.n_workers)

    def val_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size * 2, shuffle=False, num_workers=self.n_workers)


class TrainDataset(Dataset):
    def __init__(self):
        self.train = torch.load(CURRENT_PATH / 'data' / 'ml-1m-train.pt')
        self.train_pos = self.train._indices().T
        self.n_users, self.n_items = self.train.size()

        self.score = torch.sparse.sum(self.train, dim=0).to_dense().repeat((self.n_users, 1))
        self.score[self.train_pos[:, 0], self.train_pos[:, 1]] = 0

    def __len__(self):
        return self.train_pos.shape[0]

    def __getitem__(self, idx):
        return self.train_pos[idx], self.score[self.train_pos[idx][0]]


class TestDataset(Dataset):
    def __init__(self):
        self.test_pos = torch.load(CURRENT_PATH / 'data' / 'ml-1m-test-pos.pt')
        self.test_neg = torch.load(CURRENT_PATH / 'data' / 'ml-1m-test-neg.pt')
        self.n_users = self.test_pos.shape[0]

        test_items = []
        for u in range(self.n_users):
            items = torch.cat((self.test_pos[u, 1].view(1), self.test_neg[u]))
            test_items.append(items)

        self.test_items = torch.vstack(test_items)
        self.test_labels = torch.zeros(self.test_items.shape)
        self.test_labels[:, 0] += 1

    def __len__(self):
        return self.n_users

    def __getitem__(self, idx):
        return self.test_pos[idx], self.test_items[idx], self.test_labels[idx]
