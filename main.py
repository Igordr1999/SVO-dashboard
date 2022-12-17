from dash import Dash, Input, Output, dcc, html
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests

from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor

app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

dep_final = pd.read_csv("./data/dep_final.csv")
flights = pd.read_excel("./data/Расписание рейсов 05-06.2022.xlsx")
revenue_05 = pd.read_csv("./data/revenue_05.csv", parse_dates=["timeThirty"])
feature_importance_for_all_points = pd.read_csv("./data/feature_importance_for_all_points.csv")
aircraft_types = pd.read_csv("./data/aircraft_types.csv")
svo_submission = pd.read_csv("./data/svo_submission.csv")

# get key: https://account.mapbox.com/
MAPBOX_KEY = "pk.eyJ1IjoiaGliYWZpczY2MCIsImEiOiJjbGJzY2hlNnIyNG5zM3ZwcWU3d2UyZjJ6In0.NCGI1AEJa6AqR3yP1KCKbw"

all_points = ['Точка продаж 1', 'Точка продаж 2', 'Точка продаж 3',
              'Точка продаж 4', 'Точка продаж 5', 'Точка продаж 6',
              'Точка продаж 7', 'Точка продаж 8', 'Точка продаж 9',
              'Точка продаж 10', 'Точка продаж 11', 'Точка продаж 12',
              'Точка продаж 13', 'Точка продаж 14', 'Точка продаж 15',
              'Точка продаж 16', 'Точка продаж 17', 'Точка продаж 18',
              'Точка продаж 19', 'Точка продаж 20', 'Точка продаж 21',
              'Точка продаж 22', 'Точка продаж 23', 'Точка продаж 24',
              'Точка продаж 25', 'Точка продаж 26', 'Точка продаж 27',
              'Точка продаж 28', 'Точка продаж 29']


def plot_top_airlines():
    all_airlines = requests.get(
        "https://www.flightradar24.com/_json/airlines.php",
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.186 Safari/537.36 OPR/51.0.2830.55"
        }
    ).json()["rows"]
    all_airlines = pd.DataFrame(all_airlines)
    df6 = flights.AIRLINE.value_counts().reset_index(name="count")
    df6 = df6.merge(all_airlines[["Code", "Name"]], how='left', left_on='index', right_on='Code')
    df6 = df6.sort_values(by="count", ascending=False)
    df6.loc[df6['count'] < 150, 'Name'] = 'Others'
    fig6 = px.pie(df6, values='count', names='Name', title="Top airlines")
    return fig6


def plot_ac_types_pie():
    df4 = dep_final.groupby(['name_x']).size().reset_index().rename(columns={0: "count"})
    df4 = df4.sort_values(by="count", ascending=False)
    df4.loc[df4['count'] < 200, 'name_x'] = 'Others'
    fig4 = px.pie(df4, values='count', names='name_x', title="Top aircraft types (AFL + SDM)")
    return fig4


def plot_cities():
    df4 = dep_final.groupby(['arrival_scheduled_airportCode']).size().reset_index().rename(columns={0: "count"})
    df4 = df4.sort_values(by="count", ascending=False)
    df4.loc[df4['count'] < 400, 'arrival_scheduled_airportCode'] = 'Others'
    fig4 = px.pie(df4, values='count', names='arrival_scheduled_airportCode', title="Top destination airports")
    return fig4


def plot_top5_ac_types():
    top_types = flights.AIRCRAFTTYPE.value_counts()[:5].index
    df2 = flights[flights["AIRCRAFTTYPE"].isin(top_types)].groupby(
        ['FLIGHT_DATE', 'AIRCRAFTTYPE']).size().reset_index().rename(columns={0: "count"})

    return px.line(data_frame=df2,
                   x='FLIGHT_DATE',
                   y='count',
                   color='AIRCRAFTTYPE',
                   title="Top 5 aircraft types by date"
                   )


def plot_map_dest():
    px.set_mapbox_access_token(MAPBOX_KEY)

    df5 = dep_final[["name_y", "lat", "lon", 'departure_actualTakeOff']]
    df5["departure_actualTakeOff"] = pd.to_datetime(df5["departure_actualTakeOff"])
    df5['date'] = df5["departure_actualTakeOff"].dt.date
    # df5 = df5["name_y", "lat", "lon", "date"]
    df5 = df5.drop(columns=["departure_actualTakeOff"])
    df5 = df5.value_counts().reset_index(name="count")
    df5 = df5.sort_values(by='date')

    fig5 = px.scatter_mapbox(
        df5,
        lat="lat",
        lon="lon",
        color="count",
        size="count",
        hover_name="name_y",
        # color_continuous_scale=px.colors.diverging.BrBG,
        color_continuous_scale=px.colors.sequential.Viridis,
        zoom=2,
        title="Flight destinations",
        animation_frame='date',
        height=500
    )

    fig5.update_layout(margin=dict(b=0, t=0, l=0, r=0))

    return fig5


def get_feature_importance_for_point(point_name="Точка продаж 7"):
    df2 = revenue_05
    df2 = df2[df2["point"] == point_name]
    X = df2.drop(columns=['timeThirty', 'timeHour', 'revenue', 'point',
                          'date', 'datetime'
                          # 'mean_revenue'
                          ])
    y = df2['revenue']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = CatBoostRegressor(random_state=42, eval_metric="RMSE", iterations=100)
    clf.fit(X_train, y_train,
            eval_set=(X_test, y_test),
            plot=False, verbose=False
            )
    d = pd.DataFrame({
        "Point": point_name,
        "Feature name": clf.feature_names_,
        "Importance, %": clf.feature_importances_,
    })
    return d


def get_feature_importance_for_all_points():
    all_df = pd.DataFrame(columns=["Point", "Feature name", "Importance, %"])
    for point in all_points:
        df = get_feature_importance_for_point(point_name=point)
        all_df = pd.concat([all_df, df])
    return all_df


def plot_pred_vs_actual(point='Точка продаж 1'):
    chart1 = go.Figure()
    chart1.add_trace(go.Scatter(x=svo_submission[svo_submission["point"] == point].timeThirty,
                                y=svo_submission[svo_submission["point"] == point].predicted,
                                name="predicted", showlegend=True,
                                marker={"color": "tomato"},
                                mode="lines"))
    chart1.add_trace(go.Scatter(x=svo_submission[svo_submission["point"] == point].timeThirty,
                                y=svo_submission[svo_submission["point"] == point].revenue,
                                name="actual", showlegend=True,
                                marker={"color": "blue"},
                                mode="lines"))

    chart1.update_layout(height=500,
                         xaxis_title="Datetime",
                         yaxis_title="Revenue ($)",
                         title="Predicted vs actual revenue")
    return chart1


# top 15 features by point
@app.callback(
    Output("graph", "figure"),
    Input("dropdown", "value"))
def update_bar_chart(point):
    df2 = revenue_05
    df2 = df2[df2["point"] == point]
    X = df2.drop(columns=['timeThirty', 'timeHour', 'revenue', 'point',
                          'date', 'datetime'
                          # 'mean_revenue'
                          ])
    y = df2['revenue']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = CatBoostRegressor(random_state=42, eval_metric="RMSE", iterations=100)
    clf.fit(X_train, y_train,
            eval_set=(X_test, y_test),
            plot=False, verbose=False
            )

    d = pd.DataFrame({"Importance, %": clf.feature_importances_,
                      "Feature name": clf.feature_names_
                      })
    d = d.sort_values(by="Importance, %", ascending=False)[:15]

    fig = px.bar(d, x=d["Feature name"], y=d["Importance, %"], title='Impact of flights on revenue by point')
    fig.update_xaxes(tickangle=-45)
    fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
    return fig


@app.callback(
    Output("graphPoint", "figure"),
    Input("dropdownPoint", "value"))
def update_bar_chart(point):
    return plot_pred_vs_actual(point)

# points by features
@app.callback(
    Output("graphFeature", "figure"),
    Input("dropdownFeature", "value"))
def update_bar_chartFeature(feature_name):
    d = feature_importance_for_all_points[feature_importance_for_all_points["Feature name"] == feature_name]
    d = d.sort_values(by="Importance, %", ascending=False)
    fig = px.bar(d,
                 x=d["Point"],
                 y=d["Importance, %"],
                 title='Impact of flights on revenue by feature')
    fig.update_xaxes(tickangle=-45)
    fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
    return fig


chart1 = go.Figure()
chart1.update_layout(height=500,
                     xaxis_title="Date",
                     yaxis_title="Revenue ($)",
                     title="Revenue by point (date)")

graph1 = dcc.Graph(
    id='graph1',
    figure=chart1,
    # className="eight columns"
)

chart2 = go.Figure()
chart2.update_layout(height=500,
                     xaxis_title="Time",
                     yaxis_title="Revenue ($)",
                     title="Revenue by point (time)")

graph2 = dcc.Graph(
    id='graph2',
    figure=chart1,
    # className="eight columns"
)

multi_select_line_chart = dcc.Dropdown(
    id="multi_select_line_chart",
    options=[{"value": label, "label": label} for label in all_points],
    value=all_points[0:2],
    multi=True,
    clearable=False
)

row1 = html.Div(children=[multi_select_line_chart, graph1], className="eight columns")
row2 = html.Div(children=[graph2], className="eight columns")

app.layout = dbc.Container(
    [
        html.H1("SVO dashboard", style={'textAlign': 'center'}),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(children=dcc.Graph(
                    id='example-graph-1',
                    figure=plot_ac_types_pie(),
                ), md=4),
                dbc.Col(children=dcc.Graph(
                    id='example-graph-2',
                    figure=plot_top_airlines(),
                ), md=4),
                dbc.Col(children=dcc.Graph(
                    id='example-graph-3',
                    figure=plot_cities(),
                ), md=4),
                dbc.Col(children=dcc.Graph(
                    id='example-graph-4',
                    figure=plot_top5_ac_types(),
                ), md=12),

                html.P("Top destinations from Moscow", style={'textAlign': 'center'}),

                dbc.Col(children=dcc.Graph(
                    figure=plot_map_dest()
                ), md=12),

                html.Hr(),
                html.H3("Consumer activity", style={'textAlign': 'center'}),

                dbc.Col(children=row1, md=6),
                dbc.Col(children=row2, md=6),

                dbc.Col(children=dcc.Dropdown(
                    id="dropdown",
                    options=all_points,
                    value=all_points[0],
                    clearable=False,
                ), md=12),
                dbc.Col(children=dcc.Graph(id="graph"), md=12),

                dbc.Col(children=dcc.Dropdown(
                    id="dropdownFeature",
                    options=feature_importance_for_all_points["Feature name"].unique(),
                    value=feature_importance_for_all_points["Feature name"].unique()[0],
                    clearable=False,
                ), md=12),
                dbc.Col(children=dcc.Graph(id="graphFeature"), md=12),

                dbc.Col(children=dcc.Dropdown(
                    id="dropdownPoint",
                    options=all_points,
                    value=all_points[2],
                    clearable=False,
                ), md=12),
                dbc.Col(children=dcc.Graph(
                    id='graphPoint',
                    figure=plot_pred_vs_actual()))
            ],
            align="center",
        ),
    ],
    fluid=True,
)


# revenue by day
@app.callback(Output('graph1', 'figure'), [Input('multi_select_line_chart', 'value')])
def update_line1(price_options):
    revenue_05_day = revenue_05.groupby(["date", "point"])["revenue"].sum().reset_index()
    chart1 = go.Figure()

    for price_op in price_options:
        chart1.add_trace(go.Scatter(x=revenue_05_day[revenue_05["point"] == price_op].date,
                                    y=revenue_05_day[revenue_05["point"] == price_op].revenue,
                                    mode="lines", name=price_op))

    chart1.update_layout(
        xaxis_title="Date",
        yaxis_title="Revenue ($)",
        title="Revenue by point (date)",
        height=500,
    )
    return chart1


# revenue by hour
@app.callback(Output('graph2', 'figure'), [Input('multi_select_line_chart', 'value')])
def update_line2(price_options):
    revenue_05_day = revenue_05.groupby(["hour", "point"])["revenue"].sum().reset_index()
    chart2 = go.Figure()

    for price_op in price_options:
        chart2.add_trace(go.Scatter(x=revenue_05_day[revenue_05["point"] == price_op].hour,
                                    y=revenue_05_day[revenue_05["point"] == price_op].revenue,
                                    mode="lines", name=price_op))

    chart2.update_layout(
        xaxis_title="Time",
        yaxis_title="Revenue ($)",
        title="Revenue by point (time)",
        height=500,
    )
    return chart2


if __name__ == '__main__':
    app.run_server(debug=True)
