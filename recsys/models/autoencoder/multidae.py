import torch
import torch.nn as nn
import torch.nn.functional as F

from recsys.models.autoencoder.utils import MLP


class MultiDAE(nn.Module):
    def __init__(self, encoder_dims, decoder_dims, dropout=0.5):
        super().__init__()
        self.mlp = MLP(encoder_dims + decoder_dims)
        self.dropout = nn.Dropout(dropout)

    def forward(self, items):
        items_recon = self.mlp(self.dropout(items))

        return items_recon

    def loss(self, items):
        items_recon = self(items)

        return F.binary_cross_entropy_with_logits(items_recon, items)

    def negative_sampling_loss(self, items, n_negatives):
        negative_samples = items.squeeze().sum(dim=0).repeat((items.shape[0], 1))
        negative_samples[items.squeeze().nonzero(as_tuple=True)] = -1
        negative_samples = negative_samples.argsort(descending=True, dim=1)[:, :500]

        row_positives, train_positives = items.squeeze().nonzero().chunk(2, dim=1)
        row_negatives = row_positives.squeeze().repeat(n_negatives)
        col_negatives = torch.randint(
            0, negative_samples.shape[1], (row_negatives.shape[0],)
        )

        train_negatives = negative_samples[row_negatives, col_negatives].view(
            -1, n_negatives
        )
        inputs = torch.hstack([train_positives, train_negatives])
        labels = torch.zeros_like(inputs)
        labels[:, 0] = 1

        scores = self(items)
        scores = scores[row_positives.squeeze()].squeeze()
        scores = torch.take_along_dim(scores, inputs, dim=1)

        return F.binary_cross_entropy_with_logits(scores, labels.float())
