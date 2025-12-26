import pandas as pd
import matplotlib.pyplot as plt
import  seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('../data/Data_salary.csv')
def descriptive(df):
    df_num = df.select_dtypes(include=['number'])
    df_min = df_num.min()
    df_max = df_num.max()
    df_mean = df_num.mean()
    df_median = df_num.median()
    df_q1 = df_num.quantile(0.25)
    df_q2 = df_num.quantile(0.5)
    df_q3 = df_num.quantile(0.75)
    df_iqr = df_q3 - df_q1
    df_var = df_num.var()
    df_stdev = df_num.std()
    data = {
        "Min": [i for i in df_min],
        "Max": [i for i in df_max],
        "Mean": [i for i in df_mean],
        "Median": [i for i in df_median],
        "Q1": [i for i in df_q1],
        "Q2": [i for i in df_q2],
        "Q3": [i for i in df_q3],
        "IQR": [i for i in df_iqr],
        "Var": [i for i in df_var],
        "Stdev": [i for i in df_stdev]
    }
    df_data = pd.DataFrame(data)
    df_data.index = df_num.keys()
    df_complete = df_data.transpose()
    print('Bảng tóm lược mô tả dữ liệu: ')
    print(df_complete.to_string())
    return df_complete

def descriptive_table(df):
    df_num = df.select_dtypes(include=['number'])
    des_table = df_num.describe( include="all")
    print("Bảng thống kê mô tả cho các cột dữ liệu: ")
    print(des_table)
    return des_table
missing_rows_per_column = df.isnull().sum()
print(missing_rows_per_column)

descriptive_table(df)