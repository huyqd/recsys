import torch
from torch.utils.data import Dataset
from pathlib import Path

CURRENT_PATH = Path(__file__).cwd()


class ML1mDataset(Dataset):
    def __init__(self):
        self.train = torch.load(CURRENT_PATH / 'data' / 'ml-1m-train.pt')
        self.n_users, self.n_items = self.train.size()
        self.train_pos = self.train._indices().T
        self.test_pos = torch.load(CURRENT_PATH / 'data' / 'ml-1m-test-pos.pt')
        self.test_neg = torch.load(CURRENT_PATH / 'data' / 'ml-1m-test-neg.pt')

    def __len__(self):
        return self.n_users

    def __getitem__(self, idx):
        return self.train_pos[idx]
