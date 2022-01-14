import argparse
from datetime import datetime

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger
from torch import nn

from dataset import ML1mDataModule
from metrics import get_eval_metrics
from models import MODELS_DICT


class LitModule(pl.LightningModule):
    def __init__(self, model_name, n_users, n_items, embedding_dim, optim_name, lr, k=10):
        super().__init__()
        self.save_hyperparameters()

        self.model = MODELS_DICT[model_name](n_users, n_items, embedding_dim)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, users, items):
        return self.model(users, items)

    def training_step(self, batch, batch_idx):
        users, items, labels = batch

        logits = self(users, items)
        loss = self.loss(logits, labels)

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
        loss = self.loss(logits, labels)

        items = items.view(-1, n_items)
        logits = logits.view(-1, n_items)
        item_true = items[:, 0].view(-1, 1)
        item_scores = [dict(zip(item.tolist(), score.tolist())) for item, score in zip(items, logits)]
        ndcg, apak, hr = get_eval_metrics(item_scores, item_true, self.hparams.k)
        metrics = {
            'loss': loss.item(),
            'ndcg': ndcg,
            'apak': apak,
            'hr': hr,
        }

        logger.experiment.log(metrics)
        self.log("val metrics", metrics, prog_bar=True)

        return {
            "loss": loss.item(),
            "logits": logits,
        }

    def validation_epoch_end(self, outputs):
        pass

    def configure_optimizers(self):
        if self.hparams.optim_name == "AdamW":
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr)
        elif self.hparams.optim_name == "Adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        elif self.hparams.optim_name == "SGD":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.hparams.lr, momentum=0.9)
        else:
            raise NotImplementedError

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


def train_other_model(datamodule, logger, args):
    model = MODELS_DICT[args.model_name]
    logger.experiment.config['embedding_dim'] = args.embedding_dim
    logger.experiment.config['k'] = args.k
    logger.experiment.config['n_users'] = args.n_users
    logger.experiment.config['n_items'] = args.n_items

    model = model(args.embedding_dim)
    model.fit(datamodule)
    scores = model(datamodule)
    true = datamodule.test_items[:, [0]]
    ndcg, apak, hr = get_eval_metrics(scores, true, args.k)
    metrics = {
        'ndcg': ndcg,
        'apak': apak,
        'hr': hr,
    }
    logger.experiment.log(metrics)
    print(metrics)

    return model


def train_model(datamodule, logger, args):
    recommender = LitModule(args.model_name,
                            args.n_users,
                            args.n_items,
                            args.embedding_dim,
                            optim_name=args.optim,
                            lr=args.lr,
                            k=args.k)

    if logger and not (args.fast_dev_run or args.overfit_batches):
        logger.watch(recommender.model, log="all")
    else:
        logger = False

    # lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=logger,
        check_val_every_n_epoch=1,
        checkpoint_callback=False,
        num_sanity_val_steps=0,
        gradient_clip_val=1,
        gradient_clip_algorithm="norm",
        fast_dev_run=args.fast_dev_run,
        reload_dataloaders_every_n_epochs=1,  # For dynamic negative sampling
        overfit_batches=args.overfit_batches,
        gpus=1 if torch.cuda.is_available() else 0,
        # callbacks=[lr_monitor],
    )

    trainer.fit(recommender, datamodule=datamodule)

    return recommender.model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="NeuMF", help="model name")
    parser.add_argument("--k", type=int, default=10, help="k")
    parser.add_argument("--embedding-dim", type=int, default=32, help="embedding-dim")
    parser.add_argument("--n-negative-samples", type=int, default=4,
                        help="number of negative examples for neg sampling")
    parser.add_argument("--batch-size", type=int, default=1024, help="batch size for train dataloader")
    parser.add_argument("--optim", type=str, default="Adam", help="Optimizer")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--n-workers", type=int, default=4, help="number of workers for dataloader")
    parser.add_argument("--max-epochs", type=int, default=128, help="max number of epochs")
    parser.add_argument("--fast-dev-run", type=int, default=0, help="unit test")
    parser.add_argument("--overfit-batches", type=float, default=0.0, help="number of batches for overfitting purpose")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    args = parser.parse_args()

    dm = ML1mDataModule(batch_size=args.batch_size,
                        n_negative_samples=args.n_negative_samples,
                        n_workers=args.n_workers)
    args.n_users, args.n_items = dm.n_users, dm.n_items

    name = f'{args.model_name}-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    logger = WandbLogger(name=name, project="MovieLens 1M Implicit Dataset", group=args.model_name)

    if args.model_name in ("Popularity", "AlsMF"):
        train_other_model(dm, logger, args)
    else:
        # pl.seed_everything(args.seed)
        train_model(dm, logger, args)
