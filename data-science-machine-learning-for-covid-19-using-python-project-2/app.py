# =========================================================
# Imports
# =========================================================
import dash
from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
from typing import List, Tuple
import json

# =========================================================
# Constants
# =========================================================
DATE_FORMAT = "%m-%d-%Y"

GITHUB_DAILY_API = (
    "https://api.github.com/repos/CSSEGISandData/COVID-19/contents/"
    "csse_covid_19_data/csse_covid_19_daily_reports"
)

RAW_DAILY_BASE = (
    "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/"
    "csse_covid_19_data/csse_covid_19_daily_reports"
)

RAW_DAILY_US_BASE = (
    "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/"
    "csse_covid_19_data/csse_covid_19_daily_reports_us/"
)

RAW_VACCINATIONS = (
    "https://raw.githubusercontent.com/owid/covid-19-data/master/"
    "public/data/vaccinations/vaccinations.csv"
)

NEWS_ARTICLES_CSV = "covid_new_articles.csv"

# =========================================================
# Load country codes (ISO3)
# =========================================================
with open("cc3_cn_r.json") as f:
    cc3_cn_r = json.load(f)

# =========================================================
# Load United State Abbre
# =========================================================
with open("us_state_abbrev.json") as f:
    us_state_abbrev = json.load(f)



# =========================================================
# Data Loader
# =========================================================
class CovidDataLoader:
    @staticmethod
    def available_dates() -> List[datetime]:
        r = requests.get(GITHUB_DAILY_API, timeout=30)
        r.raise_for_status()
        dates = []
        for f in r.json():
            if f["name"].endswith(".csv"):
                try:
                    dates.append(
                        datetime.strptime(
                            f["name"].replace(".csv", ""), DATE_FORMAT
                        )
                    )
                except ValueError:
                    continue
        return sorted(dates)

    @staticmethod
    def load_daily(date_str: str) -> Tuple[pd.DataFrame, datetime]:
        dates = CovidDataLoader.available_dates()
        try:
            d = datetime.strptime(date_str, DATE_FORMAT)
        except Exception:
            d = dates[-1]

        if d not in dates:
            d = dates[-1]

        df = pd.read_csv(f"{RAW_DAILY_BASE}/{d.strftime(DATE_FORMAT)}.csv")
        return df, d

    @staticmethod
    def load_time_series(kind: str) -> pd.DataFrame:
        return pd.read_csv(
            "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/"
            f"csse_covid_19_data/csse_covid_19_time_series/"
            f"time_series_covid19_{kind}_global.csv"
        )
    
    @staticmethod
    def load_us_daily(date: datetime) -> pd.DataFrame:
        url = f"{RAW_DAILY_US_BASE}{date.strftime(DATE_FORMAT)}.csv"
        df = pd.read_csv(url)

        if "Province_State" not in df.columns:
            raise ValueError("Invalid US daily report format")

        return df

    @staticmethod
    def load_vaccinations() -> pd.DataFrame:
        df = pd.read_csv(RAW_VACCINATIONS)
        return df

    @staticmethod
    def load_news_articles() -> pd.DataFrame:
        return pd.read_csv(
            NEWS_ARTICLES_CSV,
            encoding="latin1"  # ou "cp1252"
        )


# =========================================================
# Services
# =========================================================
class MetricsService:
    @staticmethod
    def compute(df: pd.DataFrame) -> dict:
        confirmed = int(df["Confirmed"].sum())
        recovered = int(df["Recovered"].sum())
        deaths = int(df["Deaths"].sum())
        active = int(df["Active"].sum())
        closed = recovered + deaths

        return {
            "confirmed": confirmed,
            "recovered": recovered,
            "deaths": deaths,
            "active": active,
            "closed": closed,
            "perc_recovered": round((recovered / closed) * 100, 1) if closed else 0,
            "perc_deaths": round((deaths / closed) * 100, 1) if closed else 0,
        }

class TimeSeriesService:
    @staticmethod
    def prepare_confirmed(df):
        df = df.drop(columns=["Province/State", "Lat", "Long"], errors="ignore")
        df.set_index("Country/Region", inplace=True)
        df = df.diff(axis=1)
        df.reset_index(inplace=True)
        df = pd.melt(
            df,
            id_vars=["Country/Region"],
            var_name="date",
            value_name="value",
        )
        return df[df["value"].notna() & (df["value"] >= 0)]

    @staticmethod
    def prepare_recovered(df):
        df = df.drop(columns=["Province/State", "Lat", "Long"], errors="ignore")
        df = df.groupby("Country/Region").sum()
        df.reset_index(inplace=True)
        return pd.melt(
            df,
            id_vars=["Country/Region"],
            var_name="date",
            value_name="value",
        )

class CountryTableService:
    @staticmethod
    def build(df: pd.DataFrame) -> pd.DataFrame:
        records = []
        for country, g in df.groupby("Country_Region"):
            records.append(
                {
                    "Country_Region": country,
                    "Confirmed": int(g["Confirmed"].sum()),
                    "Deaths": int(g["Deaths"].sum()),
                    "Recovered": int(g["Recovered"].sum()),
                    "Active": int(g["Active"].sum()),
                    "Incidence_Rate": round(g["Incident_Rate"].mean(), 3)
                    if "Incident_Rate" in g else None,
                    "Case_Fatality_Ratio": round(
                        g["Case_Fatality_Ratio"].mean(), 3
                    ) if "Case_Fatality_Ratio" in g else None,
                }
            )

        df_c = pd.DataFrame(records)
        df_c["CODE"] = df_c["Country_Region"].map(cc3_cn_r)
        return df_c.dropna(subset=["CODE"])

class IndicatorFactory:
    @staticmethod
    def create(title, value, prev, color):
        fig = go.Figure(
            go.Indicator(
                mode="number+delta",
                value=value,
                title={"text": title},
                number={"valueformat": ","},
                delta={"reference": prev, "increasing": {"color": color}},
            )
        )
        fig.update_layout(height=160, margin=dict(l=10, r=10, t=40, b=10))
        return fig

class CardFactory:
    @staticmethod
    def active(m):
        return dbc.Card(
            [
                dbc.CardHeader("ACTIVE CASES"),
                dbc.CardBody(
                    [
                        html.H3(f"{m['active']:,}"),
                        html.P("Cases currently active"),
                        html.H5(f"{m['confirmed']:,}", style={"color": "green"}),
                        html.P("Total confirmed cases"),
                    ]
                ),
            ]
        )

    @staticmethod
    def closed(m):
        return dbc.Card(
            [
                dbc.CardHeader("CLOSED CASES"),
                dbc.CardBody(
                    [
                        html.H3(f"{m['closed']:,}"),
                        html.P("Cases with outcome"),
                        html.H5(
                            f"{m['recovered']:,} ({m['perc_recovered']}%)",
                            style={"color": "green"},
                        ),
                        html.H5(
                            f"{m['deaths']:,} ({m['perc_deaths']}%)",
                            style={"color": "red"},
                        ),
                    ]
                ),
            ]
        )


class USStatesService:
    @staticmethod
    def prepare(df: pd.DataFrame, state_abbrev: dict) -> pd.DataFrame:
        df = df.copy()

        state_to_code = {v: k for k, v in state_abbrev.items()}

        df["CODE"] = df["Province_State"].map(state_to_code)
        df["Confirmed"] = pd.to_numeric(df["Confirmed"], errors="coerce")

        df = df.dropna(subset=["CODE", "Confirmed"])
        df = df[df["Confirmed"] > 0]

        return df


class VaccinationService:
    @staticmethod
    def prepare(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["date"] = pd.to_datetime(df["date"])
        df["total_vaccinations"] = pd.to_numeric(
            df["total_vaccinations"], errors="coerce"
        )
        df["people_fully_vaccinated"] = pd.to_numeric(
            df["people_fully_vaccinated"], errors="coerce"
        )

        return df

    @staticmethod
    def countries(df: pd.DataFrame) -> list:
        return sorted(df["location"].dropna().unique())

class VaccinationBarService:
    @staticmethod
    def aggregate_max(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        numeric_cols = [
            "total_vaccinations",
            "people_vaccinated_per_hundred",
            "people_fully_vaccinated_per_hundred",
        ]

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df_max = (
            df.groupby("location", as_index=False)[numeric_cols]
            .max()
            .dropna(subset=["location"])
        )

        return df_max

class NewsService:
    @staticmethod
    def prepare(df: pd.DataFrame) -> list[dict]:
        required_cols = ["title","description", "image", "link"]

        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")

        return df[required_cols].to_dict("records")

class NewsCardFactory:
    @staticmethod
    def build(article: dict) -> dbc.Col:
        return dbc.Col(
            dbc.Card(
                [
                    dbc.CardImg(src=article["image"], top=True),
                    dbc.CardBody(
                        [
                            html.H6(article["title"]),
                            dbc.CardLink(
                                "Link to the Article",
                                href=article["link"],
                                target="_blank",
                            ),
                        ]
                    ),
                ]
            ),
            width=2,
        )


# =========================================================
# Initial Load
# =========================================================
df_today, date_today = CovidDataLoader.load_daily("")
df_prev, _ = CovidDataLoader.load_daily(
    (date_today - timedelta(days=1)).strftime(DATE_FORMAT)
)

metrics_today = MetricsService.compute(df_today)
metrics_prev = MetricsService.compute(df_prev)

df_ts_confirmed = TimeSeriesService.prepare_confirmed(
    CovidDataLoader.load_time_series("confirmed")
)

df_ts_recovered = TimeSeriesService.prepare_recovered(
    CovidDataLoader.load_time_series("recovered")
)

countries = sorted(
    set(df_ts_confirmed["Country/Region"])
    & set(df_ts_recovered["Country/Region"])
)

df_countries = CountryTableService.build(df_today)

df_us_states_raw = CovidDataLoader.load_us_daily(date_today)

df_us_states = USStatesService.prepare(
    df_us_states_raw,
    us_state_abbrev
)


df_vacc_raw = CovidDataLoader.load_vaccinations()
df_vacc = VaccinationService.prepare(df_vacc_raw)

countries_vacc = VaccinationService.countries(df_vacc)


df_vacc_raw = CovidDataLoader.load_vaccinations()
df_vacc_max = VaccinationBarService.aggregate_max(df_vacc_raw)

df_news_raw = CovidDataLoader.load_news_articles()
news_articles = NewsService.prepare(df_news_raw)

news_cards = [
    NewsCardFactory.build(article)
    for article in news_articles
]


# =========================================================
# WORLD MAP FIGURES
# =========================================================

fig_world_choropleth = go.Figure(
    data=go.Choropleth(
        locations=df_countries["CODE"],
        z=df_countries["Confirmed"],
        text=df_countries["Country_Region"],
        colorscale="Reds",
        autocolorscale=True,
        colorbar_title="Confirmed Cases",
    )
)

fig_world_choropleth.update_layout(
    geo=dict(
        showframe=False,
        showcoastlines=True,
        projection_type="equirectangular",
    ),
    margin=dict(l=0, r=0, t=40, b=0),
    title="World Map – Confirmed COVID-19 Cases (Choropleth)",
)

fig_world_scatter = px.scatter_geo(
    df_countries,
    locations="CODE",
    hover_name="Country_Region",
    size="Confirmed",
    projection="natural earth",
    size_max=45,
    title="World Map – Confirmed COVID-19 Cases (Scatter)",
)



# =========================================================
# WORLD MAP RADIO BUTTONS
# =========================================================
radio_buttons_world = dcc.RadioItems(
    id="radio-world-map",
    options=[
        {"label": "Scatter World Map", "value": "scatter"},
        {"label": "Choropleth World Map", "value": "choropleth"},
    ],
    value="scatter",
    labelStyle={
        "display": "inline-block",
        "marginRight": "20px"
    },
)





fig_us = go.Figure(
    go.Choropleth(
        locations=df_us_states["CODE"],
        z=df_us_states["Confirmed"],
        locationmode="USA-states",
        text=df_us_states["Province_State"],
        colorscale="Reds",
        colorbar=dict(title="Confirmed Cases"),
        zmin=df_us_states["Confirmed"].min(),
        zmax=df_us_states["Confirmed"].max(),
    )
)

fig_us.update_layout(
    title_text="COVID Cases – United States",
    geo=dict(
        scope="usa",
        projection=go.layout.geo.Projection(type="albers usa"),
    ),
    autosize=True,
    margin=dict(l=0, r=0, t=50, b=0),
)



# =========================================================
# App
# =========================================================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# =========================================================
# Layout
# =========================================================
app.layout = dbc.Container(
    [
        # NAVBAR
        dbc.Navbar(
            dbc.Container(
                dbc.NavbarBrand("CoronaMeter – COVID-19 Global Dashboard"),
                fluid=True,
            ),
            color="success",
            dark=True,
        ),

        html.Br(),

        # DATE SELECTOR
        dbc.Row(
            [
                dbc.Col(
                    dbc.Input(
                        id="date-input",
                        type="text",
                        value=date_today.strftime(DATE_FORMAT),
                        placeholder="MM-DD-YYYY",
                    ),
                    md=4,
                ),
                dbc.Col(
                    dbc.Button("Load Data", id="load-date", color="primary"),
                    md=2,
                ),
            ],
            justify="center",
        ),

        html.Br(),

        # INDICATORS
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="ind-confirmed"), md=4),
                dbc.Col(dcc.Graph(id="ind-recovered"), md=4),
                dbc.Col(dcc.Graph(id="ind-deaths"), md=4),
            ]
        ),

        html.Br(),

        # CARDS
        dbc.Row(
            [
                dbc.Col(html.Div(id="card-active"), md=6),
                dbc.Col(html.Div(id="card-closed"), md=6),
            ]
        ),

        html.Hr(),

        # DROPDOWN
        dbc.Row(
            dbc.Col(
                dcc.Dropdown(
                    id="country-dropdown",
                    options=[{"label": c, "value": c} for c in countries],
                    value=["India", "Brazil", "Australia"],
                    multi=True,
                ),
                md=8,
            ),
            justify="center",
        ),

        html.Br(),

        # GRAPHS
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="chart-confirmed"), md=6),
                dbc.Col(dcc.Graph(id="chart-recovered"), md=6),
            ]
        ),

        html.Hr(),

        # TABLE
        dbc.Row(
            dbc.Col(html.H4("COVID-19 Cases by Country"), width=10),
            justify="center",
        ),

        html.Br(),

        dbc.Row(
            dbc.Col(
                dash_table.DataTable(
                    columns=[{"name": c, "id": c} for c in df_countries.columns],
                    data=df_countries.to_dict("records"),
                    filter_action="native",
                    sort_action="native",
                    page_action="native",
                    page_size=15,
                    style_table={"overflowY": "auto"},
                    style_header={"fontWeight": "bold"},
                    style_data_conditional=[
                        {
                            "if": {"row_index": "odd"},
                            "backgroundColor": "rgb(248,248,248)",
                        }
                    ],
                ),
                width=10,
            ),
            justify="center",
        ),

        html.Br(),

        html.Hr(),

        dbc.Row(
            dbc.Col(html.H4("World COVID-19 Map"), width=10),
            justify="center",
        ),
        
        dbc.Row(
            dbc.Col(radio_buttons_world, width=10),
            justify="center",
        ),
        
        html.Br(),
        
        dbc.Row(
            dbc.Col(dcc.Graph(id="world-map-graph"), width=10),
            justify="center",
        ),

        html.Hr(),

        dbc.Row(
            dbc.Col(html.H4("United States COVID-19 Map"), width=10),
            justify="center",
        ),

        dbc.Row(
            dbc.Col(
                dcc.Graph(
                    id="us-map-graph",
                    figure=fig_us
                ),
                width=10,
            ),
            justify="center",
        ),

        html.Hr(),

        dbc.Row(
            dbc.Col(html.H4("COVID-19 Vaccination Timeline"), width=10),
            justify="center",
        ),

        dbc.Row(
            dbc.Col(
                dcc.Dropdown(
                    id="dropdown-vaccine-timeline",
                    options=[{"label": c, "value": c} for c in countries_vacc],
                    value="World",
                    clearable=False,
                ),
                width=5,
            ),
            justify="center",
        ),

        html.Br(),

        dbc.Row(
            dbc.Col(
                dcc.Graph(id="vaccine-timeline"),
                width=10,
            ),
            justify="center",
        ),

        html.Hr(),
        
        dbc.Row(
            dbc.Col(html.H4("COVID-19 Vaccination Status by Country"), width=10),
            justify="center",
        ),
        
        dbc.Row(
            dbc.Col(
                dcc.RadioItems(
                    id="radio-button-vaccine",
                    options=[
                        {
                            "label": "Total Vaccinations",
                            "value": "total_vaccinations",
                        },
                        {
                            "label": "People Vaccinated per Hundred",
                            "value": "people_vaccinated_per_hundred",
                        },
                        {
                            "label": "People Fully Vaccinated per Hundred",
                            "value": "people_fully_vaccinated_per_hundred",
                        },
                    ],
                    value="people_vaccinated_per_hundred",
                    labelStyle={"display": "inline-block", "marginRight": "15px"},
                ),
                width=10,
            ),
            justify="center",
        ),
        
        html.Br(),
        
        dbc.Row(
            dbc.Col(
                dcc.Graph(id="update-vaccine"),
                width=10,
            ),
            justify="center",
        ),

        html.Hr(),

        dbc.Row(
            dbc.Col(html.H4("Latest COVID-19 News"), width=10),
            justify="center",
        ),

        html.Br(),

        dbc.Row(
            news_cards,
            justify="center",
        ),
    ],
    fluid=True,
)

# =========================================================
# Callbacks
# =========================================================
@app.callback(
    [
        Output("ind-confirmed", "figure"),
        Output("ind-recovered", "figure"),
        Output("ind-deaths", "figure"),
        Output("card-active", "children"),
        Output("card-closed", "children"),
    ],
    Input("load-date", "n_clicks"),
    State("date-input", "value"),
)
def update_summary(_, date_str):
    df_t, d = CovidDataLoader.load_daily(date_str)
    df_p, _ = CovidDataLoader.load_daily(
        (d - timedelta(days=1)).strftime(DATE_FORMAT)
    )

    m_t = MetricsService.compute(df_t)
    m_p = MetricsService.compute(df_p)

    return (
        IndicatorFactory.create("Confirmed", m_t["confirmed"], m_p["confirmed"], "#FF4136"),
        IndicatorFactory.create("Recovered", m_t["recovered"], m_p["recovered"], "#2ECC40"),
        IndicatorFactory.create("Deaths", m_t["deaths"], m_p["deaths"], "#FF4136"),
        CardFactory.active(m_t),
        CardFactory.closed(m_t),
    )

@app.callback(
    Output("chart-confirmed", "figure"),
    Input("country-dropdown", "value"),
)
def update_confirmed(countries_sel):
    df = df_ts_confirmed[df_ts_confirmed["Country/Region"].isin(countries_sel)]
    return px.line(df, x="date", y="value", color="Country/Region",
                   title="Daily Confirmed Cases", line_shape="hv")

@app.callback(
    Output("chart-recovered", "figure"),
    Input("country-dropdown", "value"),
)
def update_recovered(countries_sel):
    df = df_ts_recovered[df_ts_recovered["Country/Region"].isin(countries_sel)]
    return px.line(df, x="date", y="value", color="Country/Region",
                   title="Total Recovered Cases (Cumulative)")

@app.callback(
    Output("world-map-graph", "figure"),
    Input("radio-world-map", "value"),
)
def update_world_map(map_type):
    if map_type == "choropleth":
        return fig_world_choropleth
    return fig_world_scatter


@app.callback(
    Output("vaccine-timeline", "figure"),
    Input("dropdown-vaccine-timeline", "value"),
)
def update_vaccine_timeline(country):
    df_f = df_vacc[df_vacc["location"] == country]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_f["date"],
            y=df_f["total_vaccinations"],
            fill="tozeroy",
            name="Total Vaccinations",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_f["date"],
            y=df_f["people_fully_vaccinated"],
            fill="tozeroy",
            name="People Fully Vaccinated",
        )
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Total Count",
        title=f"Vaccination Progress – {country}",
    )

    return fig

@app.callback(
    Output("update-vaccine", "figure"),
    Input("radio-button-vaccine", "value"),
)
def update_vaccine_bar(metric):
    fig = px.bar(
        df_vacc_max,
        x="location",
        y=metric,
        color_discrete_sequence=["green"],
        height=650,
    )

    fig.update_layout(
        xaxis_title="Country",
        yaxis_title=metric.replace("_", " ").title(),
        title="COVID-19 Vaccination Status by Country",
    )

    return fig

# =========================================================
# Run
# =========================================================
if __name__ == "__main__":
    app.run(debug=True)
