import pandas as pd
from pandas.api.types import CategoricalDtype


def create_timeseries_features(df, label=None):
    cat_type = CategoricalDtype(categories=['Monday', 'Tuesday',
                                            'Wednesday',
                                            'Thursday', 'Friday',
                                            'Saturday', 'Sunday'],
                                ordered=True)
    """
    Creates time series features from datetime index.
    """
    df = df.copy()
    df['date'] = df.index
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    # df['weekday'] = df['date'].dt.day_name()
    # df['weekday'] = df['weekday'].astype(cat_type)
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['date_offset'] = (df.date.dt.month*100 + df.date.dt.day - 320) % 1300

    #df['season'] = pd.cut(df['date_offset'], [0, 300, 602, 900, 1300],
                          #labels=['Spring', 'Summer', 'Autumn', 'Winter']
                          #)
    X = df[['hour', 'dayofweek', 'quarter', 'month', 'year',
           'dayofyear', 'dayofmonth', 'weekday',
            'season']]
    if label:
        y = df[label]
        return X, y
    return X


def denoiser(df):
    pass
