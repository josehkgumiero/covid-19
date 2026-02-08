# CoronaMeter – COVID-19 Global Dashboard

An interactive dashboard developed with Python + Dash for visualization and analysis of global COVID-19 data, integrating cases, maps, time series, vaccination data, and news into a single web application.

# Features

# Global Indicators
Confirmed cases, recovered cases, deaths, active cases, and closed cases.

# Interactive Time Series
Daily evolution of confirmed and recovered cases by country.

# World Maps

Global choropleth map (confirmed cases)

Scatter geo map by country

Toggle between map types using radio buttons

🇺🇸 United States Map

Choropleth map by state

Confirmed cases aggregated by Province_State

# Vaccination Data

Interactive vaccination timeline by country

Bar chart comparison by metric:

Total vaccinations

People vaccinated per hundred

People fully vaccinated per hundred

# COVID-19 News

Dynamic news cards with image, title, and external link

Data loaded from a local CSV file

# Project Architecture

The project follows a modular and scalable architectural pattern, improving maintainability and extensibility:

```
Constants        → URLs, formats, and static files
Data Loader      → Data ingestion (CSV, APIs, GitHub RAW)
Services         → Business rules and data preparation
Factories        → Charts, indicators, and UI components
Initial Load     → Initial data loading and preprocessing
Layout           → Application visual structure
Callbacks        → Interactivity and reactive behavior
```

# Project Structure
```
data-science-machine-learning-for-covid-19-using-python-project-2/
├── app.py
├── cc3_cn_r.json
├── us_state_abbrev.json
├── covid_new_articles.csv
├── requirements.txt
└── README.md
```

# Data Sources

Johns Hopkins CSSE

Global daily reports

Time series data

United States daily state-level data

Our World in Data (OWID)

Global vaccination dataset

Local CSV

COVID-19 related news articles

# Installation and Execution
1. Create a virtual environment
python -m venv .venv

2. Activate the virtual environment
.venv\Scripts\Activate.ps1

3. Install dependencies
pip install -r requirements.txt

4. Run the application
python app.py


The app will be available at:

http://127.0.0.1:8050

# Main Technologies

Python

Dash / Plotly

Dash Bootstrap Components

Pandas

Requests

# Technical Notes

External data loading uses GitHub RAW URLs, avoiding GitHub API rate limits.

Local CSV files are loaded using Windows-compatible encoding (latin1).

The application is fully server-side in Python, with no custom HTML or JavaScript required.

# License

This project was developed for educational and data analysis purposes.
All datasets retain their respective original licenses.