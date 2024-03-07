import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import plotly.express as px

# Function to get stock data
def get_stock_data(tickers, start_date, end_date):
    stock_data = yf.download(tickers, start=start_date, end=end_date)
    return stock_data

# Function to get tech stock returns
def get_tech_returns(tech_list, start_date, end_date):
    closing_df = yf.download(tech_list, start=start_date, end=end_date)['Adj Close']
    tech_rets = closing_df.pct_change()
    tech_rets = tech_rets.dropna()
    return tech_rets

# Read and format S&P 500 tickers
tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
symbol = dict(zip(tickers['Symbol'], tickers['Security']))
formatted_list = [{'label': f'{company} ({ticker})', 'value': ticker} for ticker, company in symbol.items()]

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the dashboard
app.layout = html.Div([
    html.H1("Stock Analysis and Forecasting Dashboard"),
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Stock Analysis', value='tab-1'),
        dcc.Tab(label='Financial Reports', value='tab-2'),
        dcc.Tab(label='Stock Price Forecasting', value='tab-3')
    ]),
    html.Div(id='tabs-content')
])

# Define callback to render tab content
@app.callback(Output('tabs-content', 'children'),
              [Input('tabs', 'value')])
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            dcc.Dropdown(
                id='stock-selector',
                options=formatted_list,
                multi=True,
                value=['AAPL', 'MSFT', 'GOOGL', 'AMZN'],  # Default selected stocks
            ),
            dcc.DatePickerRange(
                id='date-picker',
                start_date='2023-01-01',
                end_date='2024-01-01',
                display_format='YYYY-MM-DD'
            ),
            dcc.Graph(id='candlestick-subplot-1', className='subplot'),
            dcc.Graph(id='candlestick-subplot-2', className='subplot'),
            dcc.Graph(id='candlestick-subplot-3', className='subplot'),
            dcc.Graph(id='candlestick-subplot-4', className='subplot'),
            dcc.Graph(id='volume-subplot', className='subplot'),
            dcc.Graph(id='tech-scatter-plot'),
        ])
    elif tab == 'tab-2':
        return html.Div([
            dcc.Dropdown(
                id='stock-dropdown',
                options=formatted_list,
                value='MSFT',  # Default value
                clearable=False
            ),
            html.Div(id='report-output')
        ])
    elif tab == 'tab-3':
        return html.Div([
            html.Label("Select Stock Ticker:"),
            dcc.Dropdown(
                id='stock-ticker',
                options=formatted_list,
                value='AAPL'  # Default value
            ),
            dcc.Graph(id='stock-price-graph')
        ])

# Callback to update the financial report based on selected stock
@app.callback(
    Output('report-output', 'children'),
    [Input('stock-dropdown', 'value')]
)
def update_report(selected_stock):
    if selected_stock is None:
        return "Select a stock to view reports"
    else:
        # Fetch data for the selected stock
        stock_data = yf.Ticker(selected_stock)
        recommendations = stock_data.recommendations
        income_statement = stock_data.income_stmt
        balance_sheet = stock_data.balance_sheet
        cash_flow_statement = stock_data.cashflow
        major_holders = stock_data.major_holders
        institutional_holders = stock_data.institutional_holders
        mutualfund_holders = stock_data.mutualfund_holders
        insider_transactions = stock_data.insider_transactions
        insider_purchases = stock_data.insider_purchases
        insider_roster_holders = stock_data.insider_roster_holders

        # Construct the report
        report = html.Div([
            html.H3(f"Financial Report for {symbol[selected_stock]} ({selected_stock})"),
            html.H4("Recommendations for the Stock"),
            html.Pre(recommendations.to_string()),  # Convert DataFrame to string for display
            html.H4("Income Statement (Annual)"),
            html.Pre(income_statement.to_string()),
            html.H4("Balance Sheet (Annual)"),
            html.Pre(balance_sheet.to_string()),
            html.H4("Cash Flow Statement (Annual)"),
            html.Pre(cash_flow_statement.to_string()),
            html.H4("Major Holders"),
            html.Pre(major_holders.to_string()),
            html.H4("Institutional Holders"),
            html.Pre(institutional_holders.to_string()),
            html.H4("Mutual Fund Holders"),
            html.Pre(mutualfund_holders.to_string()),
            html.H4("Insider Transactions"),
            html.Pre(insider_transactions.to_string()),
            html.H4("Insider Purchases"),
            html.Pre(insider_purchases.to_string()),
            html.H4("Insider Roster Holders"),
            html.Pre(insider_roster_holders.to_string())
        ])
        return report

# Callback to update the stock price forecasting graph
@app.callback(
    Output('stock-price-graph', 'figure'),
    [Input('stock-ticker', 'value')]
)
def update_graph(selected_ticker):
    if not selected_ticker:
        # If no ticker is selected, return an empty graph
        return {'data': [], 'layout': {}}

    try:
        # Fetch data
        df = yf.download(selected_ticker, start='2012-01-01', end=datetime.now())
        data = df.filter(['Close'])
        dataset = data.values
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)
        training_data_len = int(np.ceil(len(dataset) * .95))
        train_data = scaled_data[0:int(training_data_len), :]
        x_train, y_train = [], []
        for i in range(60, len(train_data)):
            x_train.append(train_data[i-60:i, 0])
            y_train.append(train_data[i, 0])
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    except Exception as e:
        # If an error occurs, return an empty graph
        print(f"Error fetching or preprocessing data for {selected_ticker}: {e}")
        return {'data': [], 'layout': {}}

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(60, 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train LSTM model
    model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=0)

    # Prepare testing data
    test_data = scaler.transform(df.filter(['Close']).values)
    x_test = []
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Make predictions
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Create traces for the graph
    traces = []
    # True historical close prices
    traces.append(go.Scatter(
        x=df.index,
        y=df['Close'],
        mode='lines',
        name='True Historical Close Price'
    ))
    # Predicted close prices
    traces.append(go.Scatter(
        x=df.index[60:],  # Skip the first 60 points due to the window size used in LSTM
        y=predictions.flatten(),
        mode='lines',
        name='Predicted Close Price'
    ))

    # Layout of the graph
    layout = dict(
        title=f"{selected_ticker} Close Price Forecast",
        xaxis=dict(title='Date'),
        yaxis=dict(title='Close Price USD ($)'),
        showlegend=True
    )

    return {'data': traces, 'layout': layout}

# Callback to update candlestick and volume subplots
@app.callback(
    [Output('candlestick-subplot-1', 'figure'),
     Output('candlestick-subplot-2', 'figure'),
     Output('candlestick-subplot-3', 'figure'),
     Output('candlestick-subplot-4', 'figure'),
     Output('volume-subplot', 'figure'),
     Output('tech-scatter-plot', 'figure')],
    [Input('stock-selector', 'value'),
     Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date')]
)
def update_subplots_callback(selected_stocks, start_date, end_date):
    stock_data = get_stock_data(selected_stocks, start_date, end_date)
    candlestick_subplots = []

    for i, stock in enumerate(selected_stocks):
        traces = [
            {
                'x': stock_data.index,
                'open': stock_data['Open'][stock],
                'high': stock_data['High'][stock],
                'low': stock_data['Low'][stock],
                'close': stock_data['Close'][stock],
                'type': 'candlestick',
                'name': stock,
            },
            {
                'x': stock_data.index,
                'y': stock_data['Close'][stock].rolling(window=10).mean(),  # 10-day moving average
                'type': 'scatter',
                'mode': 'lines',
                'line': {'color': 'green'},
                'name': '10-day MA',
            },
            {
                'x': stock_data.index,
                'y': stock_data['Close'][stock].rolling(window=20).mean(),  # 20-day moving average
                'type': 'scatter',
                'mode': 'lines',
                'line': {'color': 'orange'},
                'name': '20-day MA',
            },
            {
                'x': stock_data.index,
                'y': stock_data['Close'][stock].rolling(window=50).mean(),  # 50-day moving average
                'type': 'scatter',
                'mode': 'lines',
                'line': {'color': 'red'},
                'name': '50-day MA',
            },
        ]

        subplot = {
            'data': traces,
            'layout': {
                'title': f'Candlestick Chart - {stock}',
                'xaxis': {'rangeslider': {'visible': False}},
                'yaxis': {'title': 'Stock Price'},
            }
        }

        candlestick_subplots.append(subplot)

    volume_trace = [
        {
            'x': stock_data.index,
            'y': stock_data['Volume'][stock],
            'type': 'bar',
            'name': stock,
        } for stock in selected_stocks
    ]

    volume_subplot = {
        'data': volume_trace,
        'layout': {
            'title': 'Trade Volume',
            'xaxis': {'rangeslider': {'visible': False}},
            'yaxis': {'title': 'Volume'},
        }
    }

    # Tech scatter plot
    tech_rets = get_tech_returns(selected_stocks, start_date, end_date)
    area = np.pi * 10
    scatter_plot = go.Figure()

    for stock in tech_rets.columns:
        scatter_plot.add_trace(
            go.Scatter(
                x=[tech_rets[stock].mean()],
                y=[tech_rets[stock].std()],
                mode='markers',
                marker=dict(size=area),
                text=[stock],
                name=stock
            )
        )

    scatter_plot.update_layout(
        title="Tech Stock Returns",
        xaxis_title="Expected Return",
        yaxis_title="Risk",
        showlegend=True
    )

    return (
        candlestick_subplots[0], candlestick_subplots[1],
        candlestick_subplots[2], candlestick_subplots[3],
        volume_subplot,
        scatter_plot
    )

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
