
RAW_DAILY_BASE = (
    "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/"
    "csse_covid_19_data/csse_covid_19_daily_reports_us'"
)


df_states = pd.read_csv(RAW_DAILY_BASE)

df_states['CODE'] = df_states['Province_State'].map(us_state_abbrev)

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