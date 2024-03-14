import numpy as np
import pandas as pd
from scipy.stats import iqr
def mean_absolute_percentage_error(y_true, y_pred):
    '''
    Mean Absolute Percentage Error (MAPE) Function
    
    y_true: list/series for actual values and predicted values
    y_pred: mape value 
    '''

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def missing_data(input_data):
    '''
    This function returns dataframe with information about the percentage of nulls in each column and the column data type.
    
    input: pandas df
    output: pandas df
    
    '''

    total = input_data.isnull().sum()
    percent = (input_data.isnull().sum()/input_data.isnull().count()*100)
    table = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    types = []
    for col in input_data.columns:
        dtype = str(input_data[col].dtype)
        types.append(dtype)
    table["Types"] = types
    return (pd.DataFrame(table))

def iqr_outlier_threshold(df, column):
    ''' Calculates the iqr outlier upper and lower thresholds

    input: pandas df, column of the same pandas df
    output: upper threshold, lower threshold

    '''
    iqr_value = iqr(df[column])
    lower_threshold = np.quantile(df[column], 0.25) - ((1.5) * (iqr_value))
    upper_threshold = np.quantile(df[column], 0.75) + ((1.5) * (iqr_value))
    print('Outlier threshold calculations:',
      f'IQR: {iqr_value}', f'Lower threshold:{lower_threshold}', f'Upper threshold: {upper_threshold}')
    
    return upper_threshold, lower_threshold