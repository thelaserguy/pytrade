import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def process_data(csv_files):
    sequence_length = 1000  # window size
    dfs_x, dfs_y = [], []
    for file in csv_files:
        df = pd.read_csv(file)
        df.drop(['Date', 'Dividends', 'Stock Splits'], axis=1, inplace=True)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        df[df.columns] = scaler.fit_transform(df[df.columns])
        # create sliding windows
        for i in range(len(df) - sequence_length):
            dfs_x.append(df[i : i + sequence_length].values)
            dfs_y.append(df.iloc[i + sequence_length]['Close'])
    return np.array(dfs_x), np.array(dfs_y)