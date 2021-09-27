import dash
from datetime import datetime as dt
import dash_html_components as html
import dash_core_components as dcc

import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from dash.dependencies  import Input,Output,State

# Building the app
app = dash.Dash(__name__)
server = app.server

# To get the graph of the stock prices
def get_stock_price_fig(df):
    fig = px.line(df,
                  x=df.Date , # Date str,
                  y=[df.Open,df.Close], # list of 'Open' and 'Close',
                  title="Closing and Opening Price vs Date")
    return fig

def get_more(df):
    df['EWA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    fig = px.scatter(df,
                    x= df.Date, # Date str,
                    y= df.EWA_20, # EWA_20 str,
                    title="Exponential Moving Average vs Date")

    fig.update_traces(mode='lines' ) # appropriate mode
    return fig



# Building the layout of the application
app.layout = html.Div(
    [   
        # First main division
        html.Div(
            [   
                html.Div(
                html.P("Welcome to Hariom's Stock Dash App!", className="header"),style={"border":"dotted"}),
                html.Br(),

                html.Div([
                    # stock code input
                    
                    html.P("Input Stock Code",className="header"),
                    html.Br(),

                    dcc.Input(id='code',placeholder="Stock Code",type="text",style={"color":"balck"}),

                    html.Button(children="Submit",id="submit-button",className="button",),
                    ]),
                
                html.Br(),

                html.Div([
                    # Date range picker input
                    dcc.DatePickerRange(id="date-picker",end_date="2021-04-01",display_format="YYYY-MM-DD"),

                    ]),
                html.Div([
                # Stock price button
                html.Button(children="Stock Price", id="stock-price-button",className="button"),
                # Indicators button
                html.Button(children="Indicators", id="indicators-button",className="button"),
                # Number of days of forecast input
                dcc.Input(id="num-days-input",placeholder="Number of Days",className=""),
                # Forecast button
                html.Button(children="Forecast", id="forecast-button",className="button"),
                    ]),
            ],
            className="inputs"
            ),
        
        # 2nd main division

        html.Div(
            [
            html.Div(#logo
            [html.Img(id="logo-link"),
            #company_name
             html.H1(id="company-name",style={"padding":"2vw","border":"center","font-size":"3vw","word-wrap":"center"}),
            ]
            , style={"display":"flex","padding":"2vw"}),

            html.Br(),

            html.Div(#Description
            [html.P(id="description", style={"padding":"2vw"})]
                , className="description_ticker"),

            html.Div(# Stock plot
                id="stock-price-plot"),

            html.Div([#indicator plot
                ],id="main-content"),
            
            html.Div([#forecast
                ],id="forecast-content")],
         style={"padding":"1vw"})

    ],className="container")

@app.callback([Output(component_id='logo-link', component_property='src'),
                Output(component_id='description', component_property='children'),
                Output(component_id="company-name",component_property="children")
                ],
                [Input(component_id='submit-button',component_property='n_clicks')],
                [State("code", "value")],
                )

def get_inforamtion(clicks,stock_name):
    if clicks is not None:
        try:
            ticker = yf.Ticker(stock_name)
            information = ticker.info
            df=pd.DataFrame().from_dict(information,orient="index").T
            #print(df.columns)
            logo_url = df['logo_url'][0]
            summary = df["longBusinessSummary"][0]
            name = df["shortName"][0]

            return [logo_url, summary, name]
        except:
            pass

@app.callback([Output(component_id="stock-price-plot",component_property='children')],
                [Input(component_id="stock-price-button",component_property="n_clicks")],
                [State(component_id="date-picker",component_property="start_date"),
                 State(component_id="date-picker",component_property="end_date"),
                 State(component_id="code",component_property="value")])
def get_prices(clicks,start_date,end_date,stock_name):
    if clicks is not None:
        try:
            df = yf.download(stock_name,start_date,end_date)
            df.reset_index(inplace=True)
            fig = get_stock_price_fig(df)
            fig.update_traces(opacity=0.5)
            figure = dcc.Graph(figure = fig)
            return [figure]       
        except:
            pass

@app.callback([Output(component_id="main-content",component_property='children')],
                [Input(component_id="indicators-button",component_property="n_clicks")],
                [State(component_id="date-picker",component_property="start_date"),
                 State(component_id="date-picker",component_property="end_date"),
                 State(component_id="code",component_property="value")])
def get_indication(clicks,start_date,end_date,stock_name):
    if clicks is not None:
        try:
            df = yf.download(stock_name,start_date,end_date)
            df.reset_index(inplace=True)
            fig = get_more(df)
            fig.update_traces(opacity=0.5)
            figure = dcc.Graph(figure = fig)
            return [figure]       
        except:
            pass




    





















if __name__=="__main__":
    app.run_server(debug=True)