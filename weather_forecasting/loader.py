# imports for loading data
import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import datetime as dt
from omegaconf import DictConfig, OmegaConf

class DataLoader():
    """
    Loads data from API for: training, fine-tuning, inference

    """  
    def __init__(self, cfg: DictConfig):
        cfg = OmegaConf.to_container(cfg, resolve=True)
        print(cfg)
        self.columns = cfg['data']['columns_to_train']
        self.begin_date_time = cfg['time']['stamps_start']
        self.lat = cfg['data']['lat_long'][0]
        self.long = cfg['data']['lat_long'][1]
        self.url_archive = cfg['urls']['url_archive']
        self.url_forecast = cfg['urls']['url_forecast']
        self.time_delta = cfg['time']['time_delta']
        self.stamps_start = cfg['time']['stamps_start']
        self.stamps_end = cfg['time']['stamps_end']
    def api_response(self, purpose:str) -> pd.DataFrame:
        """
        Configurate request to API and returns data based on request's config
        
        Args:
        purpose: for which purpose data will be used (train/fine_tune/infer)

        Returns:
        result: pd.DataFrame with data
        """
        print(self.columns)
        # making a request to API
        cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
        retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
        openmeteo = openmeteo_requests.Client(session = retry_session)
        # configurate output
        if(purpose=="train"):
            url = self.url_archive
            params = {
                "latitude": self.lat,
                "longitude": self.long,
                "start_date": self.begin_date_time,
                "end_date": (dt.datetime.today() - dt.timedelta(days = self.time_delta)).strftime('%Y-%m-%d'),
                "hourly": self.columns
            }
        elif(purpose=="infer" or purpose=="fine_tune"):
            url=self.url_forecast
            params = {
                "latitude": self.lat,
                "longitude": self.long,
                "hourly": self.columns,
                "past_days":self.time_delta
            }
        # getting data
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
        print(f"Elevation {response.Elevation()} m asl")
        print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
        print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")
        responses = openmeteo.weather_api(url, params=params)
        hourly = response.Hourly()
        hourly_data = {"date": pd.date_range(
                start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
                end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
                freq = pd.Timedelta(seconds = hourly.Interval()),
                inclusive = "left"
        )}
        for i in range(len(self.columns)):
            hourly_data[self.columns[i]] = hourly.Variables(i).ValuesAsNumpy()
        result = pd.DataFrame(data = hourly_data)
        result.dropna(inplace=True)
        if(purpose=="fine_tune"):
            result = result.iloc[:self.stamps_start,:]
        elif(purpose=="infer"):
            result = result.iloc[self.stamps_start:self.stamps_end,:]
        return result
    def load_train_data(self) -> pd.DataFrame:
        """
        Loads train data

        Returns:
        train_data: train data to load into model
        """
        train_data = self.api_response("train")
        return train_data
    
    def load_fine_tune_data(self) -> pd.DataFrame:
        """
        Loads fine-tune data

        Returns:
        fine_tune_data: fine-tune data to load into model
        """
        fine_tune_data = self.api_response("fine_tune")
        return fine_tune_data
    
    def load_infer_data(self) -> pd.DataFrame:
        """
        Loads infer data

        Returns:
        infer_data: infer data to make predictions
        """
        infer_data = self.api_response("infer")
        return infer_data
    


def main(cfg: DictConfig, purpose:str):
    data_loader = DataLoader(cfg)
    if(purpose=="train"):
        data = data_loader.load_train_data()
    elif(purpose=="fine_tine"):
        data = data_loader.load_fine_tune_data()
    elif(purpose=="infer"):
        data = data_loader.load_infer_data()
    return data
