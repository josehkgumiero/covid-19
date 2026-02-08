
radio_button_vaccine = dcc.RadioItem(
    id='radio_button_vaccine',
    options=[
        {'label': 'Total Vaccinations', 'value': 'total_vaccinations'},
        {'label': 'People Vaccinated per Hundred', 'value': 'people_vaccinated_per_hndred'},
        {'label': 'People Fully Vaccinated per Hundred', 'value': 'people_fully_vaccinated_per_hundred'},
    ], value = 'people_vaccinated_per_hundred',
    labelStyle={'display': 'inline-block'}
)


dbc.Row([dbc.Col(radio_button_vaccine, width=10)], justify='center')

@app.callback(
    Output('update-vaccine', 'figure'),
    [Input('radio_button_vaccine', 'value')])
def vaccination_status(parameter):
    fig = px.bar(df_vacc_max, x='location', y=parameter, color_discrete_sequence=['green'], \
        height=650)

    return fig

dbc.Row([dbc.Col(dcc.Graph(id='update-vaccine'), width=10)], justify='center')
