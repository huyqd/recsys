from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

CURRENT_PATH = Path(__file__).cwd()


class TrainDataset(Dataset):
    def __init__(self):
        self.train = torch.load(CURRENT_PATH / 'data' / 'ml-1m-train.pt')
        self.train_pos = self.train._indices().T
        self.n_users, self.n_items = self.train.size()

        self.score = torch.sparse.sum(self.train, dim=0).to_dense().repeat((self.n_users, 1))
        self.score[self.train_pos[:, 0], self.train_pos[:, 1]] = 0

    def __len__(self):
        return self.n_users

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


class ML1mDataset:
    def __init__(self):
        self.train_ds = TrainDataset()
        self.test_ds = TestDataset()
        self.train_dl = DataLoader(self.train_ds, batch_size=64, shuffle=True, num_workers=8)
        self.test_dl = DataLoader(self.test_ds, batch_size=128, shuffle=False, num_workers=8)
