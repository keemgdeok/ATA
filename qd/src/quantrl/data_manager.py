import os
import pandas as pd
import numpy as np

from quantrl import settings

COLUMNS_CHART_DATA = ['date', 'open', 'high', 'low', 'close', 'volume']


COLUMNS_TRAINING_DATA_V2 = ['per', 'pbr', 'roe',  
    'open_lastclose_ratio', 'high_close_ratio', 'low_close_ratio',
    'close_lastclose_ratio', 'volume_lastvolume_ratio',
    'close_ma5_ratio', 'volume_ma5_ratio',
    'close_ma10_ratio', 'volume_ma10_ratio',
    'close_ma20_ratio', 'volume_ma20_ratio',
    'close_ma60_ratio', 'volume_ma60_ratio',
    'close_ma120_ratio', 'volume_ma120_ratio',
    'market_kospi_ma5_ratio', 'market_kospi_ma20_ratio', 
    'market_kospi_ma60_ratio', 'market_kospi_ma120_ratio', 
    'bond_k3y_ma5_ratio', 'bond_k3y_ma20_ratio', 
    'bond_k3y_ma60_ratio', 'bond_k3y_ma120_ratio',
]

COLUMNS_TRAINING_DATA_V3 = COLUMNS_TRAINING_DATA_V2 + [
    'ind', 'ind_diff', 'ind_ma5', 'ind_ma10', 'ind_ma20', 'ind_ma60', 'ind_ma120',
    'inst', 'inst_diff', 'inst_ma5', 'inst_ma10', 'inst_ma20', 'inst_ma60', 'inst_ma120',
    'foreign', 'foreign_diff', 'foreign_ma5', 'foreign_ma10', 'foreign_ma20', 
    'foreign_ma60', 'foreign_ma120',
]
COLUMNS_TRAINING_DATA_V3 = list(map(
    lambda x: x if x != 'close_lastclose_ratio' else 'diffratio', COLUMNS_TRAINING_DATA_V3))


def load_data(code, date_from, date_to, ver='v2'):
    if ver in ['v3', 'v4']:
        return load_data_v3_v4(code, date_from, date_to, ver)
    elif ver in ['v4.1', 'v4.2']:
        stock_filename = ''
        market_filename = ''
        data_dir = os.path.join(settings.BASE_DIR, 'data', 'v4.1')
        for filename in os.listdir(data_dir):
            if code in filename:
                stock_filename = filename
            elif 'market' in filename:
                market_filename = filename
        
        chart_data, training_data = load_data_v4_1(
            os.path.join(data_dir, stock_filename),
            os.path.join(data_dir, market_filename),
            date_from, date_to
        )
        if ver == 'v4.1':
            return chart_data, training_data
        
        tips_filename = ''
        taylor_us_filename = ''
        data_dir = os.path.join(settings.BASE_DIR, 'data', 'v4.2')
        for filename in os.listdir(data_dir):
            if filename.startswith('tips'):
                tips_filename = filename
            if filename.startswith('taylor_us'):
                taylor_us_filename = filename
        return load_data_v4_2(
            pd.concat([chart_data, training_data], axis=1),
            os.path.join(data_dir, tips_filename),
            os.path.join(data_dir, taylor_us_filename)
        )


def load_data_v3_v4(code, date_from, date_to, ver):
    columns = None
    if ver == 'v3':
        columns = COLUMNS_TRAINING_DATA_V3

    # 시장 데이터
    df_marketfeatures = pd.read_csv(
        os.path.join(settings.BASE_DIR, 'data', ver, 'marketfeatures.csv'), 
        thousands=',', header=0, converters={'date': lambda x: str(x)})
    
    # 종목 데이터
    df_stockfeatures = None
    for filename in os.listdir(os.path.join(settings.BASE_DIR, 'data', ver)):
        if filename.startswith(code):
            df_stockfeatures = pd.read_csv(
                os.path.join(settings.BASE_DIR, 'data', ver, filename), 
                thousands=',', header=0, converters={'date': lambda x: str(x)})
            break

    # 시장 데이터와 종목 데이터 합치기
    df = pd.merge(df_stockfeatures, df_marketfeatures, on='date', how='left', suffixes=('', '_dup'))
    df = df.drop(df.filter(regex='_dup$').columns.tolist(), axis=1)

    # 날짜 오름차순 정렬
    df = df.sort_values(by='date').reset_index(drop=True)

    # NaN 처리
    df = df.fillna(method='ffill').fillna(method='bfill').reset_index(drop=True)
    df = df.fillna(0)

    # 기간 필터링
    df['date'] = df['date'].str.replace('-', '')
    df = df[(df['date'] >= date_from) & (df['date'] <= date_to)]
    df = df.reset_index(drop=True)

    # 데이터 조정
    if ver == 'v3':
        df.loc[:, ['per', 'pbr', 'roe']] = df[['per', 'pbr', 'roe']].apply(lambda x: x / 100)

    # 차트 데이터 분리
    chart_data = df[COLUMNS_CHART_DATA]

    # 학습 데이터 분리
    training_data = df[columns].values

    # 스케일링
    if ver == 'v4':
        from sklearn.preprocessing import RobustScaler
        from joblib import dump, load
        scaler_path = os.path.join(settings.BASE_DIR, 'scalers', f'scaler_{ver}.joblib')
        scaler = None
        if not os.path.exists(scaler_path):
            scaler = RobustScaler()
            scaler.fit(training_data)
            dump(scaler, scaler_path)
        else:
            scaler = load(scaler_path)
        training_data = scaler.transform(training_data)

    return chart_data, training_data