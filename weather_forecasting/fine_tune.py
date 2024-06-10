# imports for fine_tuning
import os

import hydra
import pandas as pd
import schedule
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from loader import main
from omegaconf import DictConfig
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet


def load_best_checkpoint(checkpoint_dir: str):
    """
    Load the best model checkpoint from a directory.

    Args:
    - checkpoint_dir (str): Path to the directory containing checkpoints.

    Returns:
    - model: Loaded model.
    """
    best_model_path = os.path.join(checkpoint_dir, "best-checkpoint.ckpt")

    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Best checkpoint not found at {best_model_path}")

    model = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
    return model


@hydra.main(version_base=None, config_path="../config", config_name="config")
def upload_data(cfg: DictConfig):
    """
    Upload new data.

    Returns:
    Void of nothingness
    """
    data = main(cfg, "fine_tune")
    data["id"] = "first"
    data["ind"] = [i for i in range(data.shape[0])]
    print(data)
    # костыль, потому что декорируемая гидрой функция не может возвращать значения(
    data.to_csv("fine_tune_data.csv")


def prepare_data(path_to_data: str):
    """
    Prepares new data.

    Args:
    path_to_data: path to csv file with inference data

    Returns:
    dataloader: Data to load into model
    """
    # read data from API
    data = data = pd.read_csv(path_to_data)
    data["id"] = "first" * data.shape[0]
    data["ind"] = [i for i in range(data.shape[0])]
    max_encoder_length = 36
    max_prediction_length = 6
    # create TimeSeriesDataSet for data
    dataset = TimeSeriesDataSet(
        data,
        time_idx="ind",
        target="temperature_2m",
        group_ids=["id"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        add_relative_time_idx=True,
    )

    # create DataLoader for dataset
    dataloader = dataset.to_dataloader(train=True, batch_size=64, num_workers=0)

    return dataloader


def fine_tune_model(checkpoint_dir: str):
    """
    Fine-tune the model on new data.

    Args:
    - checkpoint_dir (str): Path to the directory containing checkpoints.
    """
    # load the best trained model
    model = load_best_checkpoint(checkpoint_dir)
    upload_data()

    # prepare new data
    new_dataloader = prepare_data("fine_tune_data.csv")

    # define a checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="fine-tuned-checkpoint-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )

    # fine-tune the model
    trainer = Trainer(
        callbacks=[checkpoint_callback], max_epochs=10, accelerator="auto"
    )
    trainer.fit(model, train_dataloaders=new_dataloader)

    # save the fine-tuned model
    torch.save(
        model.state_dict(), os.path.join(checkpoint_dir, "fine_tuned_model.ckpt")
    )


schedule.every().day.at("00:00").do(fine_tune_model(checkpoint_dir="checkpoints/"))
