from __future__ import annotations
import os
from typing import Optional
import pandas as pd
from .base import HOURLY_COLUMNS, BaseSource, COL_DATETIME

class CSVSource(BaseSource):
    name = 'csv'

    def __init__(self, path: str, datetime_col: str=COL_DATETIME):
        if not os.path.isfile(path):
            raise FileNotFoundError(f'Data file not found: {path}\nRun: python scripts/fetch_istanbul_ibb.py')
        self.path = path
        self.datetime_col = datetime_col

    def load_hourly(self) -> pd.DataFrame:
        df = pd.read_csv(self.path, parse_dates=[self.datetime_col])
        if self.datetime_col != COL_DATETIME and self.datetime_col in df.columns:
            df = df.rename(columns={self.datetime_col: COL_DATETIME})
        missing = [c for c in HOURLY_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f'CSV missing columns: {missing}')
        return df[HOURLY_COLUMNS].copy()
