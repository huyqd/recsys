import argparse

import pytorch_lightning as pl
import torch
from torch import nn

from dataset import RatingML1mDataModule
from models import RATING_MODELS_DICT


class RatingLitModule(pl.LightningModule):
    def __init__(self, model_name, n_users, embedding_dim, optim_name, lr):
        super().__init__()
        self.save_hyperparameters()

        self.model = RATING_MODELS_DICT[model_name](n_users, embedding_dim)
        self.loss = nn.MSELoss()

    def forward(self, rating):
        return self.model(rating)

    def training_step(self, batch, batch_idx):
        r_hat = self(batch)
        r_hat = r_hat[batch != 0]
        batch = batch[batch != 0]
        loss = torch.sqrt(self.loss(r_hat, batch))

        return {
            "loss": loss,
        }

    def training_epoch_end(self, outputs):
        # This function recevies as parameters the output from "training_step()"
        # Outputs is a list which contains a dictionary like:
        # [{'pred':x,'target':x,'loss':x}, {'pred':x,'target':x,'loss':x}, ...]
        pass

    def validation_step(self, batch, batch_idx):
        r_hat = self(batch)
        r_hat = r_hat[batch != 0]
        batch = batch[batch != 0]
        loss = torch.sqrt(self.loss(r_hat, batch))

        metrics = {
            'rmse': loss.item(),
        }

        # logger.experiment.log(metrics)
        self.log("val metrics", metrics, prog_bar=True, on_epoch=True)

        return {
            "rmse": loss.item(),
        }

    def validation_epoch_end(self, outputs):
        pass

    def configure_optimizers(self):
        if self.hparams.optim_name == "AdamW":
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr)
        elif self.hparams.optim_name == "Adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr, weight_decay=1e-5)
        elif self.hparams.optim_name == "SGD":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.hparams.lr, momentum=0.9)
        else:
            raise NotImplementedError

        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "linear_lr_scheduler",
            }
        }


def train_model(datamodule, logger, args):
    recommender = RatingLitModule(args.model_name,
                                  datamodule.n_users,
                                  args.embedding_dim,
                                  optim_name=args.optim,
                                  lr=args.lr,
                                  )

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
        overfit_batches=args.overfit_batches,
        gpus=1 if torch.cuda.is_available() else 0,
        # callbacks=[lr_monitor],
    )

    trainer.fit(recommender, datamodule=datamodule)

    return recommender.model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="AutoEncoder", help="model name")
    parser.add_argument("--k", type=int, default=10, help="k")
    parser.add_argument("--embedding-dim", type=int, default=32, help="embedding-dim")
    parser.add_argument("--batch-size", type=int, default=128, help="batch size for train dataloader")
    parser.add_argument("--optim", type=str, default="Adam", help="Optimizer")
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    parser.add_argument("--n-workers", type=int, default=4, help="number of workers for dataloader")
    parser.add_argument("--max-epochs", type=int, default=128, help="max number of epochs")
    parser.add_argument("--fast-dev-run", type=int, default=0, help="unit test")
    parser.add_argument("--overfit-batches", type=float, default=0.0, help="number of batches for overfitting purpose")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    args = parser.parse_args()

    dm = RatingML1mDataModule(batch_size=args.batch_size, n_workers=args.n_workers)
    args.n_users, args.n_items = dm.n_users, dm.n_items

    # name = f'{args.model_name}-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    # logger = WandbLogger(name=name, project="MovieLens 1M Rating Dataset", group=args.model_name)
    logger = None

    # pl.seed_everything(args.seed)
    train_model(dm, logger, args)
