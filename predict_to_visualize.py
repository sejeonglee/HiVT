# Copyright (c) 2023, Sejeong Lee. All rights reserved.
from models.hivt_visualize import HiVTVisualize

from argparse import ArgumentParser
import functools
import pickle

import pytorch_lightning as pl
from torch_geometric.data import DataLoader

from datasets import ArgoverseV1Dataset
from utils import TemporalData

if __name__ == "__main__":
    pl.seed_everything(2022)

    parser = ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--persistent_workers", type=bool, default=True)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--pickle_prediction", type=bool, default=True)
    args = parser.parse_args()

    trainer = pl.Trainer.from_argparse_args(args)
    model = HiVTVisualize.load_from_checkpoint(
        checkpoint_path=args.ckpt_path, parallel=True
    )
    test_dataset = ArgoverseV1Dataset(
        root=args.root, split="sample", local_radius=model.hparams.local_radius
    )
    dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
    )
    predictions = functools.reduce(
        lambda former_list, latter_list: former_list + latter_list,
        trainer.predict(model, dataloader),
    )

    if args.pickle_prediction:
        filename = "predictions_for_visualize.pkl"
        print(f"Pickling predictions to {filename}")
        pickle.dump(predictions, open(filename, "wb"))
