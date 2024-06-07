# imports
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
import os
from loader import DataLoader


def prepare_data():
    """
    Prepares new data.

    Returns:
    dataloader: Data to load into model
    """
    data = DataLoader().load_infer_data()
    data["id"] = "first"
    data["ind"] = [i for i in range(data.shape[0])]
    print(data["ind"])
    max_encoder_length = 24
    max_prediction_length = 6
    dataset = TimeSeriesDataSet(
        data,
        time_idx= "ind",  
        target= "temperature_2m",  
        group_ids=["id"],  
        max_encoder_length=max_encoder_length,  
        max_prediction_length=max_prediction_length,  
        min_encoder_length=10,
        min_prediction_length=5,
        add_relative_time_idx = True,
    )
    
    dataloader = dataset.to_dataloader(train=False, batch_size=64, num_workers=0)
    
    return dataloader

def load_fine_tuned_checkpoint(checkpoint_dir: str):
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

def make_predictions(model, dataloader):
    """
    Make predictions using the loaded model and a DataLoader.
    
    Args:
    - model: The trained model.
    - dataloader: DataLoader for the dataset to make predictions on.
    
    Returns:
    - predictions: The predictions made by the model.
    """

    predictions = model.predict(dataloader).cpu().numpy()[0]

    print(predictions)
    
    
    return predictions

data = prepare_data()

model = load_fine_tuned_checkpoint("checkpoints/")

predictions = make_predictions(model, data)
print(predictions)