from __future__ import annotations
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from typing import Iterable, List, Optional
from urllib.parse import urlencode
import pandas as pd
from .. import config
from .base import COL_DATETIME, COL_NO2, COL_PM10, COL_PM25, COL_PM25_ESTIMATED, COL_SOURCE, COL_STATION_ID, COL_STATION_NAME, HOURLY_COLUMNS, BaseSource

class IBBSource(BaseSource):
    name = 'ibb_istanbul'

    def __init__(self, start: Optional[datetime]=None, end: Optional[datetime]=None, station_ids: Optional[Iterable[str]]=None, skip_station_names: Optional[Iterable[str]]=None, pm10_to_pm25_factor: float=config.PM10_TO_PM25_FACTOR, request_pause_sec: float=0.15):
        self.start = start or config.ISTANBUL_START_DATE
        self.end = end or config.ISTANBUL_END_DATE
        self.station_ids = list(station_ids) if station_ids else None
        self.skip_station_names = {s.lower() for s in skip_station_names or ['Mobil']}
        self.pm10_to_pm25_factor = pm10_to_pm25_factor
        self.request_pause_sec = request_pause_sec

    def fetch_stations(self) -> List[dict]:
        raw = self._http_get(config.IBB_STATIONS_ENDPOINT)
        return json.loads(raw)

    def load_hourly(self) -> pd.DataFrame:
        stations = self.fetch_stations()
        if self.station_ids:
            id_set = set(self.station_ids)
            stations = [s for s in stations if s.get('Id') in id_set]
        else:
            stations = [s for s in stations if str(s.get('Name', '')).lower() not in self.skip_station_names]
        frames: List[pd.DataFrame] = []
        for (idx, station) in enumerate(stations, start=1):
            sid = station['Id']
            sname = station.get('Name', sid)
            print(f'  [IBB] Station {idx}/{len(stations)}: {sname}', flush=True)
            chunk_frames = []
            for (chunk_start, chunk_end) in _six_month_chunks(self.start, self.end):
                rows = self._fetch_station_chunk(sid, sname, chunk_start, chunk_end)
                if rows:
                    chunk_frames.append(pd.DataFrame(rows))
                time.sleep(self.request_pause_sec)
            if chunk_frames:
                part = pd.concat(chunk_frames, ignore_index=True)
                frames.append(part)
                print(f'    -> {len(part):,} rows total for station', flush=True)
        if not frames:
            return pd.DataFrame(columns=HOURLY_COLUMNS)
        df = pd.concat(frames, ignore_index=True)
        df = df.sort_values([COL_STATION_ID, COL_DATETIME]).reset_index(drop=True)
        return df

    def save_hourly_csv(self, path: str) -> pd.DataFrame:
        df = self.load_hourly()
        config.ensure_data_dir()
        df.to_csv(path, index=False)
        return df

    def _fetch_station_chunk(self, station_id: str, station_name: str, start: datetime, end: datetime) -> List[dict]:
        params = {'StationID': station_id, 'StartDate': start.strftime(config.IBB_DATETIME_FORMAT), 'EndDate': end.strftime(config.IBB_DATETIME_FORMAT)}
        url = f'{config.IBB_MEASUREMENTS_ENDPOINT}?{urlencode(params)}'
        try:
            raw = self._http_get(url)
        except urllib.error.HTTPError as exc:
            print(f'    ! HTTP error for {station_name}: {exc}')
            return []
        except urllib.error.URLError as exc:
            print(f'    ! URL error for {station_name}: {exc}')
            return []
        text = raw.decode('utf-8', errors='replace').strip()
        if not text.startswith('['):
            if 'FormatException' in text or 'sorun' in text.lower():
                print(f'    ! API error for {station_name}: {text[:120]}…')
            return []
        records = json.loads(text)
        return [self._normalize_record(r, station_id, station_name) for r in records]

    @staticmethod
    def _http_get(url: str, timeout: int=90) -> bytes:
        req = urllib.request.Request(url, headers={'User-Agent': 'BLU513E-AQI-FIS/1.0'})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()

    def _normalize_record(self, record: dict, station_id: str, station_name: str) -> dict:
        conc = record.get('Concentration') or {}
        aqi_block = record.get('AQI') or {}
        pm10 = _safe_float(conc.get('PM10'))
        no2 = _safe_float(conc.get('NO2'))
        pm25 = pm10 * self.pm10_to_pm25_factor if pm10 is not None else None
        ibb_aqi = _safe_float(aqi_block.get('AQIIndex'))
        row = {COL_DATETIME: pd.to_datetime(record.get('ReadTime')), COL_STATION_ID: station_id, COL_STATION_NAME: station_name, COL_PM10: pm10, COL_PM25: pm25, COL_NO2: no2, COL_PM25_ESTIMATED: True, COL_SOURCE: self.name}
        if ibb_aqi is not None:
            row['ibb_aqi_index'] = ibb_aqi
        return row

def _safe_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(value)
        if v < 0 or v != v:
            return None
        return v
    except (TypeError, ValueError):
        return None

def _six_month_chunks(start: datetime, end: datetime):
    period_start = start
    while period_start <= end:
        month = period_start.month + 6
        year = period_start.year + (month - 1) // 12
        month = (month - 1) % 12 + 1
        try:
            next_start = period_start.replace(year=year, month=month)
        except ValueError:
            next_start = period_start.replace(year=year, month=month, day=28)
        chunk_end = min(end, next_start - timedelta(seconds=1))
        yield (period_start, chunk_end)
        period_start = chunk_end + timedelta(seconds=1)
