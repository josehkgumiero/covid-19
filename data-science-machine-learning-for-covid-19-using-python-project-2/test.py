import pandas as pd

import json


RAW_DAILY_US_BASE = (
    "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/"
    "csse_covid_19_data/csse_covid_19_daily_reports_us/01-01-2021.csv"
)

df_states = pd.read_csv(RAW_DAILY_US_BASE)

print(df_states)
with open("us_state_abbrev.json") as f:
    us_state_abbrev = json.load(f)

state_to_code = {v: k for k, v in us_state_abbrev.items()}


df_states['CODE'] = df_states['Province_State'].map(state_to_code)

print(df_states['CODE'] )

fig_us = go.Figure(data=go.Choropleth(
    locations=df_states['CODE'],
    z= def_states['Confirmed'], 
    locationmode='USA-states',
    text = df_state['Province_State'],
    colorscale='Reds',
    colorbar_title='Confirmed_Cases'
))

fig_us.update_layout(
    title_text='COVID Cases USA',
    geo_scape='usa',
    autosize=True
)

dbc.Row([dbc.Col(dbc.Graph(figure=fig_usa, width=10))], justify='center')