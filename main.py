from warnings import simplefilter
from utils.data_loader import DataLoader
from src.method_one import MethodOne


def pre_init():
    simplefilter(action='ignore', category=FutureWarning)
    simplefilter(action='ignore', category=DeprecationWarning)


if __name__ == '__main__':
    pre_init()

    dataloader = DataLoader()
    # df = dataloader.load("cern.us.txt")
    df = dataloader.load("aapl.us.txt")
    # df = dataloader.load("aapl.us.txt")
    # dataloader.show_OHLC(df)
    # dataloader.show_volume(df)
    # dataloader.show_decomposition(df)

    method_one = MethodOne(df)
    method_one.feature_engineering(False)
    method_one.split_data(False)
    # method_one.find_best_params()
    method_one.train()
    method_one.predict(True)
