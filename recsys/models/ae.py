from torch import nn


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Linear(input_dim, embedding_dim)
        self.decoder = nn.Linear(embedding_dim, input_dim)
        self.dropout = nn.Dropout(p=0.05)
        self.g = nn.Sigmoid()
        self.f = nn.Identity()

    def forward(self, x):
        return self.f(self.decoder(self.dropout(self.g(self.encoder(x)))))
