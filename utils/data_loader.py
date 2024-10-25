import pandas as pd
from stldecompose import decompose

from utils.timer import timer
import plotly.graph_objs as go


class DataLoader:
    load_directory = ""

    def __init__(self, load_directory="data/Stocks/"):
        self.load_directory = load_directory

    @timer
    def load(self, file_name, since_year=2000):
        df = pd.read_csv(self.load_directory + file_name)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df[(df['Date'].dt.year >= since_year)].copy()
        df.index = range(len(df))
        return df

    def show_OHLC(self, df):
        fig = go.Figure(data=go.Ohlc(x=df.Date,
                                     open=df.Open,
                                     high=df.High,
                                     low=df.Low,
                                     close=df.Close,
                                     name='Price'))
        fig.update_layout(title='OHLC')
        fig.show()

    def show_volume(self, df):
        fig = go.Figure(data=go.Scatter(x=df.Date, y=df.Volume, name='Volume'))
        fig.update_layout(title='Volume')
        fig.show()

    def show_decomposition(self, df):
        df_close = df[['Date', 'Close']].copy()
        df_close = df_close.set_index('Date')

        decomp = decompose(df_close, period=365)
        fig = decomp.plot()
        fig.show()
