import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb

def create_model(data, n_estimators):
    tss = TimeSeriesSplit(n_splits=5, test_size=24*365*1, gap=24)
    data = data.sort_index()

    fold = 0
    preds = []
    scores = []

    def create_features(df):
        df = df.copy()
        df['hour'] = df.index.hour
        df['month'] = df.index.month
        df['year'] = df.index.year

        return df
        
    data = create_features(data)

    def add_lags(df):
        target_map = df['Energy Use (MW)'].to_dict()
        df['1 year'] = (df.index - pd.Timedelta('364 days')).map(target_map)
        df['2 years'] = (df.index - pd.Timedelta('728 days')).map(target_map)
        df['3 years'] = (df.index - pd.Timedelta('1092 days')).map(target_map)    

        return df
        
    data = add_lags(data)
        
    for train_index, val_index in tss.split(data):
        train = data.iloc[train_index]
        test = data.iloc[val_index]
        
        train = create_features(train)
        test = create_features(test)

        FEATURES = ['hour', 'month', 'year', '1 year', '2 years', '3 years']
        TARGET = 'Energy Use (MW)'

        x_train = train[FEATURES]
        y_train = train[TARGET]

        x_test = test[FEATURES]
        y_test = test[TARGET]

        reg = xgb.XGBRegressor(
            n_estimators=n_estimators, 
            early_stopping_rounds=50, 
            learning_rate=0.01,
            objective='reg:squarederror',
            max_depth=1,
            base_score=0.5)
        reg = reg.fit(x_train, y_train,
            eval_set=[(x_train, y_train), (x_test, y_test)],
            verbose=100)
        
        y_pred = reg.predict(x_test)
        preds.append(y_pred)
        scores.append(scores)
        fold += 1

    test['prediction'] = reg.predict(x_test)
    data = data.merge(test[['prediction']],
                      how='left',
                      left_index=True,
                      right_index=True)
    
    return reg, data, preds, scores
