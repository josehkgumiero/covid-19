# COVID-19 Symptom Checker Chatbot (Rasa + Python)

This project is a conversational AI chatbot built with **Rasa** and **Python**, designed to estimate the probability of COVID-19 infection based on user-reported symptoms and exposure information.

The assistant collects structured inputs through a form and calculates a probability score using a rule-based scoring system.

---

# Project Overview

This chatbot:

- Collects symptom data (cold, fever, cough)
- Asks about recent travel
- Evaluates isolation level
- Calculates a probability score (0%–100%)
- Returns a diagnostic estimate

The project follows the standard Rasa architecture created using:

rasa init


---

#  Project Architecture
```
data-science-machine-learning-for-covid-19-using-python-project-4/
│
├── actions/ # Custom Python actions
├── data/ # Training data (NLU, rules, stories)
│ ├── nlu.yml
│ ├── rules.yml
│ ├── stories.yml
│ ├── raw/
│ ├── processed/
│ └── storage/
│
├── models/ # Trained models
├── tests/ # Conversation tests
├── config.yml # NLU and pipeline configuration
├── domain.yml # Intents, slots, responses, forms
├── endpoints.yml # Action server configuration
├── credentials.yml # Messaging platform credentials
├── requirements.txt # Python dependencies
└── README.md
```

---

# How It Works

The chatbot operates in three main layers:

## 1 NLU (Natural Language Understanding)
Defined in `data/nlu.yml`

Responsible for:
- Detecting user intents
- Extracting entities
- Mapping answers to slots

---

## 2 Dialogue Management
Defined in:
- `rules.yml`
- `stories.yml`
- `domain.yml`

Responsible for:
- Managing conversation flow
- Triggering forms
- Handling fallback scenarios

---

## 3 Custom Actions (Python Logic)
Defined in:

actions/actions.py


### Key Components

### `ValidateCovidForm`
- Controls the order of questions
- Validates user input
- Ensures isolation level is between 1 and 5

### `ActionFinalScore`
- Retrieves collected slot values
- Calculates a probability score based on:
  - Cold
  - Fever
  - Cough
  - Travel history
  - Isolation level
- Returns a percentage estimation

The final score is calculated using weighted symptom contributions and isolation risk logic.

---

# Scoring Logic

Each symptom contributes to the final score:

| Factor            | Weight |
|-------------------|--------|
| Cold              | 20%    |
| Fever             | 20%    |
| Cough             | 20%    |
| Travel            | 20%    |
| Isolation Level   | Up to 20% |

The final probability is normalized between 0% and 100%.

---



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

# Commands

- Initializes a new Rasa project
```
rasa init
```
- Trains only the NLU (Natural Language Understanding) model
```
rasa train nlu
```
- Starts an interactive shell to test only the NLU model
```
rasa shell nlu
```

- Stops the current conversation in the Rasa shell
```
/stop
```

- Validates the training data and domain configuration files
```
rasa data validate
```

- Trains the full Rasa model,
```
rasa train
```

- Starts the Rasa server to run the trained assistant
```
rasa run
```
- Starts the Rasa shell in debug mode,
```
rasa shell --debug
```
- Starts the action server, which executes custom actions defined in actions.py.
```
rasa run actions
```









```
Example Conversation
User:

I have fever and cough.

Bot:

Have you traveled recently?

Bot:

On a scale from 1 to 5, how isolated are you?

Final Response:

There is an estimated 60% probability that you may have COVID-19.
```

# Technologies Used
Python 3.10+

Rasa Open Source

Rasa SDK

YAML

Rule-based scoring logic

Form validation

# Purpose of the Project
This project demonstrates:

Conversational AI development

Form-based slot filling

Custom action integration

Validation logic

Structured dialogue management

End-to-end chatbot architecture

# Disclaimer
This chatbot is for educational purposes only and does not replace medical diagnosis.

# Author
Developed as part of a Machine Learning and Conversational AI project using Rasa and Python
