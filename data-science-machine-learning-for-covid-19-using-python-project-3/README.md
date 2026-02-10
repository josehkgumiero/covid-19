# Tweet classification - COVID-19 Tweets

# Problem Statement

- Ever since the pandemic outbreak, there has been a lot of talk on Social Media about things related to the virus
- More than 600 million tweets about Coronavirus and COVID-19 since the 1st of January of 2020.
- Can we build a model classify these tweets in covid-19 and non-covid-19?


# About the dataset

- The dataset we will be using was a part of a Hackton organized by zindi.Africa sponsored by Microsoft.

- One unique and challeging aspect of the dataset is:
    - Tweets have been classified as covid-19 (1) or not covid-19-related (0).
    - All tweets have had the following keywords removed:
        - corona
        - coronavirus
        - covid
        - covid19
        - covid-19
        - sarscov2
        - 19

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

#### Register venv like kernel jupyter
```
python -m ipykernel install --user --name covid19-venv-python-3-11 --display-name "Python 3.11 (covid-19)"
```