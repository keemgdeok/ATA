import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# from quantrl 
import settings

COLUMNS = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

def load_data(code):
    # 데이터 불러오기
    #df = pd.read_csv('time_series_data.csv', parse_dates=['date'], index_col='date')
    #f.dropna(df, inplace=True)

    for filename in os.listdir(os.path.join(settings.BASE_DIR, 'data')):
        if filename.startswith(code):
            df = pd.read_csv(
                os.path.join(settings.BASE_DIR, 'data', filename), 
                thousands=',', header=0,  converters={'Date': lambda x: str(x)}, index_col=0)
            break
    print(f'inital df shape: {df.shape}')
    # NaN 값 처리
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # 데이터셋 분리
    chart_df = df.loc[:, COLUMNS]
    train_df = df.drop(columns=COLUMNS)

    # 스케일링 수행
    scaler = StandardScaler()
    train_df[train_df.columns] = scaler.fit_transform(train_df)

    print(f"chart_df shape: {chart_df.shape}")
    print(f"train_df shape: {train_df.shape}")
    
    return chart_df, train_df

#load_data('005930')
