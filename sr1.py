import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def loadD(file_name):
    return pd.read_excel(file_name)

def VisualD(data):
    return sns.pairplot(data)

if __name__ == "__main__":
    data = loadD('C:\\Users\\Mkbv2\\OneDrive\\Documents\\python_ai_assessment\\Net_Worth_Data.xlsx')
    VisualD(data)
    plt.show()