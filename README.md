# Weather and road collisions in Great Britain, 2025

An interactive Streamlit dashboard showing how daily rainfall and temperature relate to the number and severity of road collisions in Great Britain in 2025.

Each DfT-recorded collision is matched to the Met Office HadUK-Grid rainfall and temperature in its 5 km grid cell on the day it happened. Collisions are compared with what place, season, weekday and road type would predict.

## Run locally

```bash
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```

## Files

| File | Contents |
|---|---|
| `dashboard/app.py` | The Streamlit dashboard |
| `dashboard/exposure_counts.npz` | Days each 5 km cell spent in each weather band, by month and weekday |
| `data/merged/collisions_weather_2025.csv` | One row per collision, with matched weather |
| `data/merged/daily_gb_weather_collisions_2025.csv` | One row per day: Great Britain weather and collision counts |

## Data sources

- Met Office HadUK-Grid gridded climate observations, 5 km daily rainfall and mean air temperature, 2025
- Department for Transport, Road Safety Data: collisions and casualties, 2025
- Office for National Statistics, Countries (December 2025) Boundaries UK BFC

Contains public sector information licensed under the Open Government Licence v3.0.

Developed for the Big Data Visualisation module (COS7046-B), Coursework 1.
