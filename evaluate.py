import argparse

import pytorch_lightning as pl

from dataset import ML1mDataModule
from metrics import get_eval_metrics
from models import *
from utils import Engine

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=10, help="k")
    parser.add_argument("--embedding-dim", type=int, default=32, help="embedding-dim")
    parser.add_argument("--n-negative-samples", type=int, default=4,
                        help="number of negative examples for neg sampling")
    parser.add_argument("--batch-size", type=int, default=1024, help="batch size for train dataloader")
    parser.add_argument("--n-workers", type=int, default=8, help="number of workers for dataloader")
    parser.add_argument("--max-epochs", type=int, default=200, help="max number of epochs")
    parser.add_argument("--fast-dev-run", type=int, default=0, help="unit test")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    args = parser.parse_args()
    model = AlsMF

    dm = ML1mDataModule(batch_size=args.batch_size,
                        n_negative_samples=args.n_negative_samples,
                        n_workers=args.n_workers)
    dm.setup()
    n_users, n_items = dm.n_users, dm.n_items

    if model in (Popularity, AlsMF):
        model = model(args.embedding_dim)
        model.fit(dm)
        scores = model(dm)
        true = dm.test_items[:, [0]]
        ndcg, apak, hr = get_eval_metrics(scores, true, args.k)
        metrics = {
            'ndcg': ndcg,
            'apak': apak,
            'hr': hr,
        }
        print(metrics)
    else:
        pl.seed_everything(args.seed)
        model = model(n_users, n_items, args.embedding_dim)
        recommender = Engine(model, k=args.k)
        # lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
        trainer = pl.Trainer(
            max_epochs=args.max_epochs,
            logger=False,
            check_val_every_n_epoch=1,
            checkpoint_callback=False,
            num_sanity_val_steps=0,
            gradient_clip_val=1,
            gradient_clip_algorithm="norm",
            fast_dev_run=args.fast_dev_run,
            reload_dataloaders_every_n_epochs=args.max_epochs // 10,  # For dynamic negative sampling
            # callbacks=[lr_monitor],
        )

        trainer.fit(recommender, datamodule=dm)
