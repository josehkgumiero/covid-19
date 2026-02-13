# COVID-19 Time Series Forecasting with Prophet

A complete Machine Learning project for COVID-19 time series analysis and forecasting using Python, Pandas, and Meta Prophet.

This project follows professional Data Engineering and Machine Learning architecture standards, organizing the code into modular layers inside the src/ directory.

# Project Objective

The goal of this project is to:

Ingest public COVID-19 data

Transform cumulative data into structured time series

Apply forecasting using Prophet

Generate future predictions

Visualize results (static and interactive)

Maintain clean, scalable, and modular architecture

This project is designed as a professional-level portfolio example of applied time series forecasting.

# Project Architecture
```
data-science-machine-learning-for-covid-19-using-python-project-6/
│
├── notebooks/
│   └── COVID-19 Time Series Analysis.ipynb
│
├── src/
│   ├── ingestion/
│   ├── transformation/
│   ├── modeling/
│   ├── visualization/
│   └── __init__.py
│
├── _setup.py
├── requirements.txt
└── README.md
```

# Project Layers
# Ingestion Layer

Responsible for:

Loading public COVID-19 CSV data

Managing external data access

Keeping data acquisition isolated

Main file:

src/ingestion/covid_data_loader.py

# Transformation Layer

Responsible for:

Filtering specific countries

Converting wide format → long format

Converting cumulative cases → daily cases

Filtering by date ranges

Preparing data in Prophet format (ds, y)

Files:

src/transformation/covid_transformer.py
src/transformation/time_series_transformer.py

# Modeling Layer

Responsible for:

Initializing Prophet model

Adding country holidays

Training the model

Temporal train/test split

Generating future forecasts

Predicting on existing datasets

File:

src/modeling/prophet_model.py

# Visualization Layer

Responsible for:

Static plotting (Matplotlib)

Interactive plotting (Plotly)

Forecast visualization

Model components visualization (trend, seasonality, holidays)

Files:

src/visualization/prophet_visualizer.py
src/visualization/time_series_visualizer.py

# Execution Pipeline

The full workflow:
```
Ingestion
   ↓
Transformation
   ↓
Feature Engineering
   ↓
Temporal Split
   ↓
Model Training
   ↓
Forecast Generation
   ↓
Visualization
```

This structure ensures separation of responsibilities and scalability.

# Technologies Used

Python 3.10+

Pandas

NumPy

Matplotlib

Seaborn

Prophet (Meta)

Plotly

Logging


# Installation dependencies
1. Create a virtual environment
```
python -m venv .venv
```

2. Activate the virtual environment
```
.venv\Scripts\Activate.ps1
```

3. Install dependencies
```
pip install -r requirements.txt
```

# Register venv like kernel jupyter
```
python -m ipykernel install --user --name covid19-venv-time-series-project --display-name "COVID-19 Time Series Project"
```

# What the Model Does

Prophet models time series using:

y(t) = trend + seasonality + holidays


The model:

Learns historical growth trends

Detects weekly and yearly seasonality

Incorporates national holidays

Generates future forecasts with confidence intervals

# Output

The model produces:

Historical observations

Estimated trend line

Confidence intervals

Future projections

Decomposed components (trend, weekly, yearly, holidays)

# Technical Highlights

✔ Modular architecture
✔ Separation of concerns
✔ Object-Oriented Programming
✔ Logging and error handling
✔ Scalable structure
✔ Industry-standard design
✔ Portfolio-ready implementation

# Possible Future Improvements

Automated evaluation metrics (MAE, RMSE)

Rolling window backtesting

Hyperparameter tuning

CI/CD integration

API deployment

Docker containerization

Interactive dashboard application

MLOps pipeline integration

# Author

Developed as a professional Machine Learning engineering project focused on clean architecture and forecasting best practices.

# License

Educational and demonstration purposes.