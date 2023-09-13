from pythonai import LOADingDATA, DATAOLD, DATASPLITING
import pandas as pd
import numpy as np

#Test case to ensure the correct loading of data
def test_LOADingDATA():
    data = LOADingDATA('C:\\Users\\Mkbv2\\OneDrive\\Documents\\python_ai_assessment\\test\\Net_Worth_Data.xlsx')
    assert isinstance(data, pd.DataFrame), "Loaded data should be a DataFrame."

    expected_cols = ['Client Name', 'Client e-mail', 'Profession', 'Education', 
                        'Country', 'Gender', 'Age', 'Income','Credit Card Debt','Healthcare Cost','Inherited Amount',
                        'Stocks','Bonds','Mutual Funds','ETFs','REITs']

    for col in expected_cols:
        assert col in data.columns, f"Expected column {col} not found in loaded data."

#Test case to ensure the correct range of data
def test_data_range():
    data = LOADingDATA('C:\\Users\\Mkbv2\\OneDrive\\Documents\\python_ai_assessment\\test\\Net_Worth_Data.xlsx')
    X_scaled, y_scaled, _, _ = DATAOLD(data)
    
    assert 0 <= np.min(X_scaled) <= 1, "X data should be scaled between 0 and 1."
    assert 0 <= np.min(y_scaled) <= 1, "Y data should be scaled between 0 and 1."
    assert 0 <= np.max(X_scaled) <= 1, "X data should be scaled between 0 and 1."
    assert 0 <= np.max(y_scaled) <= 1, "Y data should be scaled between 0 and 1."

#Test case to ensure the correct splitting of data
def test_DATASPLITING():
    data = LOADingDATA('C:\\Users\\Mkbv2\\OneDrive\\Documents\\python_ai_assessment\\test\\Net_Worth_Data.xlsx')
    X, Y, _, _ = DATAOLD(data)
    X_train, X_test, y_train, y_test = DATASPLITING(X, Y)
    # Check proportions for train-test split
    assert X_train.shape[0] / X.shape[0] == 0.8
    assert X_test.shape[0] / X.shape[0] == 0.2
    assert y_train.shape[0] / Y.shape[0] == 0.8
    assert y_test.shape[0] / Y.shape[0] == 0.2