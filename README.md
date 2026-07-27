# Mamdani FIS for Air Quality Index Estimation (Istanbul)

Python code accompanying the study:

**Air Quality Index Estimation Using Mamdani Fuzzy Inference System: A Case Study on Istanbul Urban Air Quality Data**

**Authors:** Anil Cagin Gok, Gulcihan Ozdemir  
**Affiliation:** Informatics Institute, Istanbul Technical University  
**Corresponding author:** goka26@itu.edu.tr

## Repository contents

| Path | Description |
|------|-------------|
| `fuzzy_aqi/` | Mamdani FIS, calibration, evaluation, baselines, IBB data source |
| `main.py` | Synthetic / controlled experiments |
| `main_real_data.py` | Istanbul IBB real-data pipeline |
| `main_extended.py` | Extended benchmarks and comparisons |
| `requirements.txt` | Python dependencies |
| `data/` | IBB station metadata, hourly and daily tables |
| `results/` | Numeric experiment outputs (CSV / JSON / TXT) |

## Data

- `data/istanbul_stations.json` — station metadata  
- `data/istanbul_ibb_hourly.csv` — cleaned hourly records from the IBB Open Data Portal (stored with Git LFS)  
- `data/istanbul_ibb_daily.csv` — daily station-level table derived from the hourly file  

Raw public source: [IBB Open Data Portal](https://data.ibb.gov.tr/).

## Setup

```bash
# If cloning: install Git LFS first, then
#   git lfs install
git clone https://github.com/AnilCaginGok/Mamdani-FIS-AQI-Istanbul.git
cd Mamdani-FIS-AQI-Istanbul

python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
```

```bash
python main.py
python main_real_data.py
python main_extended.py
```
