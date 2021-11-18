import pytorch_lightning as pl
import torch
from torch import nn

from metrics import get_eval_metrics


class Engine(pl.LightningModule):
    def __init__(self, model, k=10):
        super().__init__()
        self.k = k
        self.save_hyperparameters()

        self.model = model

    def forward(self, users, items):
        return self.model(users, items)

    def training_step(self, batch, batch_idx):
        users, items, labels = batch

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
        users, items, labels = batch
        n_items = items.shape[1]

        users = users.view(-1, 1).squeeze()
        items = items.view(-1, 1).squeeze()
        labels = labels.view(-1, 1).squeeze()

        logits = self(users, items)
        loss = self.loss_fn(logits, labels)

        items = items.view(-1, n_items)
        logits = logits.view(-1, n_items)
        item_true = items[:, 0].view(-1, 1)
        item_scores = [dict(zip(item.tolist(), score.tolist())) for item, score in zip(items, logits)]
        ndcg, apak, hr = get_eval_metrics(item_scores, item_true, self.k)
        metrics = {
            'loss': loss.item(),
            'ndcg': ndcg,
            'apak': apak,
            'hr': hr,
        }
        self.log("val metrics", metrics, prog_bar=True)

        return {
            "loss": loss.item(),
            "logits": logits,
        }

    def validation_epoch_end(self, outputs):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.001)
        n_steps = self.trainer.max_epochs * len(self.train_dataloader())
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,
                                                         start_factor=1,
                                                         end_factor=0,
                                                         total_iters=n_steps)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "linear_lr_scheduler",
            }
        }

    def loss_fn(self, logits, labels):
        return nn.BCEWithLogitsLoss()(logits, labels)
