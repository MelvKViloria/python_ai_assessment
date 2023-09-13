import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from joblib import dump, load
import numpy as np

class FactMOD:
    @staticmethod
    def GETMOD(mod):
        if mod == 'Linear Regression':
            return LinearRegression()
        elif mod == 'Support Vector Machine':
            return SVR()
        elif mod == 'Random Forest':
            return RandomForestRegressor()
        elif mod == 'Gradient Boosting Regressor':
            return GradientBoostingRegressor()
        elif mod == 'XGBRegressor':
            return XGBRegressor()
        elif mod == 'Ridge Regression':
            return Ridge()
        elif mod =='Lasso Regression':
            return Lasso()
        else:
            raise ValueError(f"Model '{mod}' not recognized!")

def LOADingDATA(file_name):
    return pd.read_excel('C:\\Users\\Mkbv2\\OneDrive\\Documents\\python_ai_assessment\\test\\Net_Worth_Data.xlsx')

def DATAOLD(DATA):
# Check for missing values
    if DATA.isnull().any().any():
        raise ValueError("The data contains missing values. Please ensure the data is cleaned before processing.")

    x = DATA.drop(['Client Name', 'Client e-mail', 'Country', 'Net Worth','Profession','Education','Healthcare Cost','Gender','Net Worth'], axis=1)
    y = DATA['Net Worth']
    
    MINMAXSCALE = MinMaxScaler()
    X_S = MINMAXSCALE.fit_transform(x)
    
    MINMAX1 = MinMaxScaler()
    ReshapeY = y.values.reshape(-1, 1)
    Y_S = MINMAX1.fit_transform(ReshapeY)
    
    return X_S, Y_S, MINMAXSCALE, MINMAX1

def DATASPLITING(X_S, Y_S):
    return train_test_split(X_S, Y_S, test_size=0.2, random_state=42)

def Train_MODS(Training_X, Training_Y):
    mod_ns = [
        'Linear Regression',
        'Support Vector Machine',
        'Random Forest',
        'Gradient Boosting Regressor',
        'XGBRegressor',
        'Ridge Regression',
        'Lasso Regression'

    ]
    
    MODS = {}
    for name in mod_ns:
        #Display model name
        print(f"Training model: {name}")
        MOD = FactMOD.GETMOD(name)
        MOD.fit(Training_X, Training_Y.ravel())
        MODS[name] = MOD
        #display when model trained successfully
        print(f"{name} trained successfully.")
        
    return MODS 


def evaluate_MODS(MODS, test_X, test_Y):
    rmsev = {}
    
    for name, MOD in MODS.items():
        PREDS = MOD.predict(test_X)
        rmsev[name] = mean_squared_error(test_Y, PREDS, squared=False)
        
    return rmsev

def PLOT_MOD_PERFORMANCE(rmsev):
    plt.figure(figsize=(10,7))
    MODS = list(rmsev.keys())
    rmse = list(rmsev.values())
    bars = plt.bar(MODS, rmse, color=['blue', 'green', 'red', 'purple', 'orange','pink','yellow'])

    for bar in bars:
        Y_VAL = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, Y_VAL + 0.00001, round(Y_VAL, 5), ha='center', va='bottom', fontsize=10)

    plt.xlabel('Models')
    plt.ylabel('Root Mean Squared Error (rmse)')
    plt.title('Model rmse Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# rmse = best mean squared error

def BM_SAVE(MODS, rmsev):
    BM = min(rmsev, key=rmsev.get)  
    BM = MODS[BM]   
    dump(BM, "Networth1.4.joblib")

def NEW_DATA_PREDICT(LOADED_MOD, MINMAXSCALE, MINMAX1):
    networthtest = MINMAXSCALE.transform(np.array([[42,62812.09301,11609.38091,35321.45877,75661.97242,42870.8743,39218.97651,58483.05364,642453.5727]]))
    PREDICTEDV = LOADED_MOD.predict(networthtest)
    print(PREDICTEDV)
    
    # Ensure PREDICTEDV is a 2D array before inverse transform
    if len(PREDICTEDV.shape) == 1:
        PREDICTEDV = PREDICTEDV.reshape(-1, 1)

    print("Predicted output: ", MINMAX1.inverse_transform(PREDICTEDV))

if __name__ == "__main__":
    try: #add try except to handle missing value error
        DATA = LOADingDATA('C:\\Users\\Mkbv2\\OneDrive\\Documents\\python_ai_assessment\\test\\Net_Worth_Data.xlsx')
        X_S, Y_S, MINMAXSCALE, MINMAX1 = DATAOLD(DATA)
        Training_X, test_X, Training_Y, test_Y = DATASPLITING(X_S, Y_S)
        MODS = Train_MODS(Training_X, Training_Y)
        rmsev = evaluate_MODS(MODS, test_X, test_Y)
        PLOT_MOD_PERFORMANCE(rmsev)
        BM_SAVE(MODS, rmsev)
        LOADED_MOD = load("Networth.joblib")
        NEW_DATA_PREDICT(LOADED_MOD, MINMAXSCALE, MINMAX1)
    except ValueError as ve:
        print(f"Error: {ve}")

