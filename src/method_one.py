import pandas as pd
import plotly.graph_objs as go
import plotly.subplots as subplots
import xgboost as xgb
from sklearn.svm import SVR
from xgboost import plot_importance
from sklearn.model_selection import GridSearchCV

from utils.timer import timer


class MethodOne:
    df = None

    train_df = None
    valid_df = None
    test_df = None

    y_train = None
    X_train = None
    y_valid = None
    X_valid = None
    y_test = None
    X_test = None

    model = None
    best_params = {'gamma': 0.02, 'learning_rate': 0.05, 'max_depth': 8, 'n_estimators': 300, 'random_state': 42}

    def __init__(self, df):
        self.df = df

    @timer
    def feature_engineering(self, show=False):
        self.calculate_MA(show)
        self.calculate_RSI(show)
        self.calculate_MACD(show)
        self.data_clean()

    def calculate_MA(self, show=False):
        self.df['EMA_9'] = self.df['Close'].ewm(9).mean().shift()
        self.df['SMA_5'] = self.df['Close'].rolling(5).mean().shift()
        self.df['SMA_10'] = self.df['Close'].rolling(10).mean().shift()
        self.df['SMA_15'] = self.df['Close'].rolling(15).mean().shift()
        self.df['SMA_30'] = self.df['Close'].rolling(30).mean().shift()

        if show:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=self.df.Date, y=self.df.EMA_9, name='EMA 9'))
            fig.add_trace(go.Scatter(x=self.df.Date, y=self.df.SMA_5, name='SMA 5'))
            fig.add_trace(go.Scatter(x=self.df.Date, y=self.df.SMA_10, name='SMA 10'))
            fig.add_trace(go.Scatter(x=self.df.Date, y=self.df.SMA_15, name='SMA 15'))
            fig.add_trace(go.Scatter(x=self.df.Date, y=self.df.SMA_30, name='SMA 30'))
            fig.add_trace(go.Scatter(x=self.df.Date, y=self.df.Close, name='Close', opacity=0.2))
            fig.show()

    def calculate_RSI(self, show=False, n=14):
        close = self.df['Close']
        delta = close.diff()[1:]
        pricesUp = delta.copy()
        pricesDown = delta.copy()
        pricesUp[pricesUp < 0] = 0
        pricesDown[pricesDown > 0] = 0
        rollUp = pricesUp.rolling(n).mean()
        rollDown = pricesDown.abs().rolling(n).mean()
        rs = rollUp / rollDown
        self.df['RSI'] = 100.0 - (100.0 / (1.0 + rs))
        self.df['RSI'].fillna(0, inplace=True)

        if show:
            fig = go.Figure(go.Scatter(x=self.df.Date, y=self.df.RSI, name='RSI'))
            fig.show()

    def calculate_MACD(self, show=False):
        EMA_12 = pd.Series(self.df['Close'].ewm(span=12, min_periods=12).mean())
        EMA_26 = pd.Series(self.df['Close'].ewm(span=26, min_periods=26).mean())
        self.df['MACD'] = pd.Series(EMA_12 - EMA_26)
        self.df['MACD_signal'] = pd.Series(self.df.MACD.ewm(span=9, min_periods=9).mean())

        if show:
            fig = subplots.make_subplots(rows=2, cols=1)
            fig.add_trace(go.Scatter(x=self.df.Date, y=self.df.Close, name='Close'), row=1, col=1)
            fig.add_trace(go.Scatter(x=self.df.Date, y=EMA_12, name='EMA 12'), row=1, col=1)
            fig.add_trace(go.Scatter(x=self.df.Date, y=EMA_26, name='EMA 26'), row=1, col=1)
            fig.add_trace(go.Scatter(x=self.df.Date, y=self.df['MACD'], name='MACD'), row=2, col=1)
            fig.add_trace(go.Scatter(x=self.df.Date, y=self.df['MACD_signal'], name='Signal line'), row=2, col=1)
            fig.show()

    def data_clean(self):
        self.df['Close'] = self.df['Close'].shift(-1)  # 把未来一天的闭盘数值作为标签
        self.df = self.df.iloc[33:]  # 在MA和MACD操作中进行了范围平均的操作，所以前面一部分数据没有相关数据，需要截去
        self.df = self.df[:-1]  # 在上面进行了shift -1 的操作，所以最后一个数据没有Close，所以需要截去最后一位
        self.df.index = range(len(self.df))

    @timer
    def split_data(self, show=False):
        test_size = 0.15
        valid_size = 0.15

        test_split_idx = int(self.df.shape[0] * (1 - test_size))
        valid_split_idx = int(self.df.shape[0] * (1 - (valid_size + test_size)))

        self.train_df = self.df.loc[:valid_split_idx].copy()
        self.valid_df = self.df.loc[valid_split_idx + 1:test_split_idx].copy()
        self.test_df = self.df.loc[test_split_idx + 1:].copy()

        if show:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=self.train_df.Date, y=self.train_df.Close, name='Training'))
            fig.add_trace(go.Scatter(x=self.valid_df.Date, y=self.valid_df.Close, name='Validation'))
            fig.add_trace(go.Scatter(x=self.test_df.Date, y=self.test_df.Close, name='Test'))
            fig.show()

        drop_cols = ['Date', 'Volume', 'Open', 'Low', 'High', 'OpenInt']
        self.train_df = self.train_df.drop(drop_cols, 1)
        self.valid_df = self.valid_df.drop(drop_cols, 1)
        self.test_df = self.test_df.drop(drop_cols, 1)

        self.y_train = self.train_df['Close'].copy()
        self.X_train = self.train_df.drop(['Close'], 1)

        self.y_valid = self.valid_df['Close'].copy()
        self.X_valid = self.valid_df.drop(['Close'], 1)

        self.y_test = self.test_df['Close'].copy()
        self.X_test = self.test_df.drop(['Close'], 1)

    @timer
    def find_best_params(self):
        parameters = {
            'n_estimators': [100, 200, 300, 400],
            'learning_rate': [0.001, 0.005, 0.01, 0.05],
            'max_depth': [8, 10, 12, 15],
            'gamma': [0.001, 0.005, 0.01, 0.02],
            'random_state': [1024]
        }
        eval_set = [(self.X_train, self.y_train), (self.X_valid, self.y_valid)]
        self.model = xgb.XGBRegressor(objective='reg:squarederror')
        clf = GridSearchCV(self.model, parameters)
        clf.fit(self.X_train, self.y_train, eval_set=eval_set, verbose=False)
        self.best_params = clf.best_params_
        print(f'Best params: {clf.best_params_}')
        print(f'Best validation score = {clf.best_score_}')

    @timer
    def train(self, params=None):
        if params is None:
            params = self.best_params

        eval_set = [(self.X_train, self.y_train), (self.X_valid, self.y_valid)]
        self.model = xgb.XGBRegressor(**params, objective='reg:squarederror')
        self.model.fit(self.X_train, self.y_train, eval_set=eval_set, verbose=False)
        # self.model = SVR(kernel="rbf")
        # self.model.fit(self.X_train, self.y_train)
        # plot_importance(self.model)

    def predict(self, show=False):
        y_pred = self.model.predict(self.X_test)
        test_split_idx = int(self.df.shape[0] * (1 - 0.15))
        predicted_prices = self.df.loc[test_split_idx + 1:].copy()
        predicted_prices['Close'] = y_pred

        if show:
            fig = subplots.make_subplots(rows=2, cols=1)
            fig.add_trace(go.Scatter(x=self.df.Date, y=self.df.Close,
                                     name='Truth',
                                     marker_color='LightSkyBlue'), row=1, col=1)

            fig.add_trace(go.Scatter(x=predicted_prices.Date,
                                     y=predicted_prices.Close,
                                     name='Prediction',
                                     marker_color='MediumPurple'), row=1, col=1)

            fig.add_trace(go.Scatter(x=predicted_prices.Date,
                                     y=self.y_test,
                                     name='Truth',
                                     marker_color='LightSkyBlue',
                                     showlegend=False), row=2, col=1)

            fig.add_trace(go.Scatter(x=predicted_prices.Date,
                                     y=y_pred,
                                     name='Prediction',
                                     marker_color='MediumPurple',
                                     showlegend=False), row=2, col=1)

            fig.show()
