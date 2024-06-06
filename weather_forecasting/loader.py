# imports for loading data
import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import datetime as dt

# constant values
CONST_DELTA = dt.timedelta(days = 7)
CONST_STAMPS_START = 144
CONST_STAMPS_END = 168
CONST_URL_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
CONST_URL_FORECAST = "https://api.open-meteo.com/v1/forecast"
CONST_COLS = ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature", "precipitation", "rain", 
            "snowfall", "direct_radiation", "sunshine_duration", "wind_speed_10m", "wind_direction_10m"]
CONST_LAT_LONG = [52.52, 13.41]
CONST_START_DATE = "2020-02-06"

class DataLoader():
    """
    Loads data from API for: training, fine-tuning, inference

    """
    def __init__(self) -> object: 
        self.columns = CONST_COLS
        self.begin_date_time = CONST_START_DATE
        self.lat = CONST_LAT_LONG[0]
        self.long = CONST_LAT_LONG[1]
    def api_response(self, purpose:str) -> pd.DataFrame:
        """
        Configurate request to API and returns data based on request's config
        
        Args:
        purpose: for which purpose data will be used (train/fine_tune/infer)

        Returns:
        result: pd.DataFrame with data
        """
        # making a request to API
        cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
        retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
        openmeteo = openmeteo_requests.Client(session = retry_session)
        # configurate output
        if(purpose=="train"):
            url = CONST_URL_ARCHIVE
            params = {
                "latitude": self.lat,
                "longitude": self.long,
                "start_date": self.begin_date_time,
                "end_date": (dt.datetime.today() - CONST_DELTA).strftime('%Y-%m-%d'),
                "hourly": self.columns
            }
        elif(purpose=="infer" or purpose=="fine_tune"):
            url=CONST_URL_FORECAST
            params = {
                "latitude": self.lat,
                "longitude": self.long,
                "hourly": self.columns,
                "past_days":7
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
            result = result.iloc[:CONST_STAMPS_START,:]
        elif(purpose=="infer"):
            result = result.iloc[CONST_STAMPS_START:CONST_STAMPS_END,:]
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