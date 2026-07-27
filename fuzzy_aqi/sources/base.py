from abc import ABC, abstractmethod
import pandas as pd
COL_DATETIME = 'datetime'
COL_STATION_ID = 'station_id'
COL_STATION_NAME = 'station_name'
COL_PM25 = 'pm25'
COL_PM10 = 'pm10'
COL_NO2 = 'no2'
COL_PM25_ESTIMATED = 'pm25_estimated'
COL_SOURCE = 'source'
HOURLY_COLUMNS = [COL_DATETIME, COL_STATION_ID, COL_STATION_NAME, COL_PM25, COL_PM10, COL_NO2, COL_PM25_ESTIMATED, COL_SOURCE]

class BaseSource(ABC):
    name: str = 'base'

    @abstractmethod
    def load_hourly(self) -> pd.DataFrame:
        ...

    def describe(self) -> str:
        return f'{self.__class__.__name__}({self.name})'
