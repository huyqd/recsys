import pytorch_lightning as pl
import torch
from torch import nn

from metrics import get_eval_metrics


class Engine(pl.LightningModule):
    def __init__(self, model, n_negative_samples, k=10):
        super().__init__()
        self.model = model
        self.n_negative_samples = n_negative_samples
        self.k = k

    def forward(self, users, items):
        return self.model(users, items)

    def training_step(self, batch, batch_idx):
        pos, score = batch
        users, pos_items = pos[:, 0], pos[:, 1]

        neg_items = torch.multinomial(score, self.n_negative_samples)
        items = torch.cat((pos_items.view(-1, 1), neg_items), dim=1)

        labels = torch.zeros(items.shape)
        labels[:, 0] += 1
        users = users.view(-1, 1).repeat(1, items.shape[1])

        users = users.view(-1, 1).squeeze()
        items = items.view(-1, 1).squeeze()
        labels = labels.view(-1, 1).squeeze()

        logits = self(users, items)
        loss = self.loss_fn(logits, labels)

        return {
            "loss": loss,
            "logits": logits.detach(),
        }

    def training_epoch_end(self, outputs):
        # This function recevies as parameters the output from "training_step()"
        # Outputs is a list which contains a dictionary like:
        # [{'pred':x,'target':x,'loss':x}, {'pred':x,'target':x,'loss':x}, ...]
        pass

    def validation_step(self, batch, batch_idx):
        pos, items, labels = batch
        n_items = items.shape[1]
        users = pos[:, 0].view(-1, 1).repeat(1, n_items)

        users = users.view(-1, 1).squeeze()
        items = items.view(-1, 1).squeeze()
        labels = labels.view(-1, 1).squeeze()

        logits = self(users, items)
        loss = self.loss_fn(logits, labels)

        items = items.view(-1, n_items)
        logits = logits.view(-1, n_items)
        item_true = pos[:, 1].view(-1, 1)
        item_scores = [dict(zip(item.tolist(), score.tolist())) for item, score in zip(items, logits)]
        ncdg, apak, hr = get_eval_metrics(item_scores, item_true, self.k)
        metrics = {
            'loss': loss.item(),
            'ncdg': ncdg,
            'apak': apak,
            'hr': hr,
        }
        self.log("Val Metrics", metrics, prog_bar=True)

        return {
            "loss": loss.item(),
            "logits": logits,
        }

    def validation_epoch_end(self, outputs):
        pass

    # def configure_optimizers(self):
    #     optimizer = torch.optim.SGD(self.model.parameters(), lr=0.05)
    #     return optimizer
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.005)
        return optimizer

    def loss_fn(self, logits, labels):
        return nn.BCEWithLogitsLoss()(logits, labels)
