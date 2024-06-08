# imports for loading data
import os.path

import hydra
# imports for training
import lightning.pytorch as pl
import pandas as pd
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.tuner import Tuner
from loader import DataLoader, main
from omegaconf import DictConfig
from pytorch_forecasting import (QuantileLoss, TemporalFusionTransformer,
                                 TimeSeriesDataSet)


@hydra.main(version_base=None, config_path="../config", config_name="config")
def train_model(cfg: DictConfig):
    # loading data
    data_path = os.path.join(os.path.dirname(__file__), "..", "data.csv")
    if os.path.isfile(data_path):
        data = pd.read_csv(data_path)
    else:
        load = main(cfg, "train")
        data = load.load_train_data()

    # add classes for series and range of stamps
    data["id"] = "first" * data.shape[0]
    data["ind"] = [i for i in range(data.shape[0])]

    # define encoder and prediction length
    max_encoder_length = 24
    max_prediction_length = 6

    # convert to TimeSeriesDataSet
    training = TimeSeriesDataSet(
        data,
        time_idx="ind",
        target="temperature_2m",
        group_ids=["id"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        add_relative_time_idx=True,
    )

    # create validation dataset
    validation = TimeSeriesDataSet.from_dataset(
        training,
        data,
        min_prediction_idx=training.index.time.max() + 1,
        stop_randomization=True,
    )

    # convert datasets to dataloaders for training
    batch_size = 128
    train_dataloader = training.to_dataloader(
        train=True, batch_size=batch_size, num_workers=2
    )
    val_dataloader = validation.to_dataloader(
        train=False, batch_size=batch_size, num_workers=2
    )

    # checkpoint callback for saving checkpoints of model
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints/",
        filename="best-checkpoint",
        save_top_k=1,
        mode="min",
    )
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="gpu",
        devices=-1,
        gradient_clip_val=0.1,
        limit_train_batches=30,  # 30 batches per epoch
        callbacks=[checkpoint_callback],
        logger=TensorBoardLogger("lightning_logs"),
    )

    # define network to train - the architecture is mostly inferred from the dataset, so that only a few hyperparameters have to be set by the user
    tft = TemporalFusionTransformer.from_dataset(
        # dataset
        training,
        # architecture hyperparameters
        hidden_size=32,
        attention_head_size=1,
        dropout=0.1,
        hidden_continuous_size=16,
        # loss metric to optimize
        loss=QuantileLoss(),
        # logging frequency
        log_interval=2,
        # optimizer parameters
        learning_rate=0.03,
        reduce_on_plateau_patience=4,
    )
    print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

    # find the optimal learning rate
    res = Tuner(trainer).lr_find(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        early_stop_threshold=1000.0,
        max_lr=0.3,
        attr_name="learning_rate",
    )
    print(f"suggested learning rate: {res.suggestion()}")

    # fit the model on the data - redefine the model with the correct learning rate if necessary
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )


train_model()
