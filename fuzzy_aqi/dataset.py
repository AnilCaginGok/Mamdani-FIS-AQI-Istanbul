from __future__ import annotations
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from . import config
from .sources.base import COL_DATETIME, COL_NO2, COL_PM10, COL_PM25, COL_PM25_ESTIMATED, COL_STATION_ID, BaseSource

def filter_valid_hourly(df: pd.DataFrame, particulate_mode: str='pm25') -> pd.DataFrame:
    if particulate_mode == 'pm10':
        (p_col, p_min, p_max) = (COL_PM10, config.PM10_MIN, config.PM10_MAX)
    else:
        (p_col, p_min, p_max) = (COL_PM25, config.PM25_MIN, config.PM25_MAX)
    mask = df[p_col].notna() & df[COL_NO2].notna() & (df[p_col] >= p_min) & (df[p_col] <= p_max) & (df[COL_NO2] >= config.NO2_MIN) & (df[COL_NO2] <= config.NO2_MAX)
    return df.loc[mask].copy()

def hourly_to_daily(df: pd.DataFrame, particulate_mode: str='pm25') -> pd.DataFrame:
    if particulate_mode == 'pm10':
        (p_col, out_col) = (COL_PM10, 'pm10')
    else:
        (p_col, out_col) = (COL_PM25, 'pm25')
    df = df.copy()
    df['date'] = df[COL_DATETIME].dt.floor('D')
    agg_dict = {out_col: (p_col, 'mean'), 'no2': (COL_NO2, 'max'), 'n_hours': (p_col, 'count')}
    if COL_PM25_ESTIMATED in df.columns:
        agg_dict['pm25_estimated'] = (COL_PM25_ESTIMATED, 'max')
    if COL_PM10 in df.columns and out_col != 'pm10':
        agg_dict['pm10'] = (COL_PM10, 'mean')
    if COL_PM25 in df.columns and out_col != 'pm25':
        agg_dict['pm25'] = (COL_PM25, 'mean')
    daily = df.groupby([COL_STATION_ID, 'date'], as_index=False).agg(**agg_dict)
    daily = daily.rename(columns={'date': COL_DATETIME})
    return daily

def time_split(df: pd.DataFrame, train_ratio: float=config.TRAIN_RATIO, val_ratio: float=config.VAL_RATIO) -> Dict[str, pd.DataFrame]:
    df = df.sort_values(COL_DATETIME).reset_index(drop=True)
    n = len(df)
    if n == 0:
        return {'train': df, 'val': df.iloc[0:0], 'test': df.iloc[0:0]}
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return {'train': df.iloc[:train_end].copy(), 'val': df.iloc[train_end:val_end].copy(), 'test': df.iloc[val_end:].copy()}

class AirQualityDataset:

    def __init__(self, source: BaseSource, use_daily: bool=True, particulate_mode: str=None):
        self.source = source
        self.use_daily = use_daily
        self.particulate_mode = particulate_mode or config.ISTANBUL_PARTICULATE_MODE
        self.particulate_col = 'pm10' if self.particulate_mode == 'pm10' else 'pm25'
        self.hourly: Optional[pd.DataFrame] = None
        self.daily: Optional[pd.DataFrame] = None
        self.splits: Optional[Dict[str, pd.DataFrame]] = None

    def build(self) -> 'AirQualityDataset':
        hourly = self.source.load_hourly()
        hourly = filter_valid_hourly(hourly, self.particulate_mode)
        self.hourly = hourly
        if self.use_daily:
            self.daily = hourly_to_daily(hourly, self.particulate_mode)
            self.splits = time_split(self.daily)
        else:
            self.daily = None
            self.splits = time_split(hourly)
        return self

    def evaluation_frame(self, split: str='test') -> pd.DataFrame:
        if self.splits is None:
            raise RuntimeError('Call build() before accessing splits.')
        return self.splits[split]

    def summary(self) -> str:
        if self.hourly is None:
            return 'Dataset not built yet.'
        lines = [f'Source     : {self.source.describe()}', f'Mode       : {self.particulate_mode.upper()} + NO2 (EPA-aligned)', f'Hourly rows: {len(self.hourly):,}', f'Stations   : {self.hourly[COL_STATION_ID].nunique()}', f'Date range : {self.hourly[COL_DATETIME].min()} to {self.hourly[COL_DATETIME].max()}']
        if self.daily is not None:
            lines.append(f'Daily rows : {len(self.daily):,}')
        if self.splits:
            for (name, part) in self.splits.items():
                lines.append(f'  {name:5s}: {len(part):,} rows')
        if self.particulate_mode == 'pm10':
            lines.append('Note       : Native PM10 from IBB (no conversion)')
        else:
            est = self.hourly[COL_PM25_ESTIMATED].any() if COL_PM25_ESTIMATED in self.hourly.columns else False
            if est:
                lines.append(f'Note       : PM2.5 estimated from PM10 (factor={config.PM10_TO_PM25_FACTOR})')
        return '\n'.join(lines)
