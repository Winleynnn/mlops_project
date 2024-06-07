# imports for fine_tuning
import torch
from lightning.pytorch import Trainer
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
import os
from lightning.pytorch.callbacks import ModelCheckpoint
from loader import DataLoader
import schedule

def load_best_checkpoint(checkpoint_dir: str):
    """
    Load the best model checkpoint from a directory.
    
    Args:
    - checkpoint_dir (str): Path to the directory containing checkpoints.
    
    Returns:
    - model: Loaded model.
    """
    best_model_path = os.path.join(checkpoint_dir, 'best-checkpoint.ckpt')
    
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Best checkpoint not found at {best_model_path}")
    
    model = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
    return model

def prepare_data():
    """
    Prepares new data.

    Returns:
    dataloader: Data to load into model
    """
    # read data from API
    data = DataLoader().load_fine_tune_data()
    data["id"] = "first"*data.shape[0]
    data["ind"] = [i for i in range(data.shape[0])]
    max_encoder_length = 36
    max_prediction_length = 6
    # create TimeSeriesDataSet for data
    dataset = TimeSeriesDataSet(
        data,
        time_idx= "ind",  
        target= "temperature_2m",  
        group_ids=["id"],  
        max_encoder_length=max_encoder_length,  
        max_prediction_length=max_prediction_length,  
        add_relative_time_idx = True
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
    model = load_best_checkpoint(checkpoint_dir)\
    
    # prepare new data
    new_dataloader = prepare_data()

    # define a checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='fine-tuned-checkpoint-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        monitor='val_loss',
        mode='min'
    )

    # fine-tune the model
    trainer = Trainer(
                      callbacks=[checkpoint_callback], 
                      max_epochs=10,
                      accelerator="gpu",  
                      devices = -1
                      )
    trainer.fit(model, train_dataloaders=new_dataloader)
    
    # save the fine-tuned model
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'fine_tuned_model.ckpt'))

schedule.every().day.at("00:00").do(fine_tune_model(checkpoint_dir="checkpoints/"))