import pandas as pd
import warnings

warnings.filterwarnings("ignore")

import os
print(os.getcwd())
os.chdir('C:\\Users\\Jelena\\PycharmProjects\\pythonProjects')



df_traintest = pd.read_csv('SolarPowerPrediction/rnn_article/traintest_features_ds.csv')


df_original = pd.read_csv('SolarPowerPrediction/rnn_article/traintest_data.csv')

import pandas as pd
import numpy as np


def generate_basic_statistics(df):
   
    # Numerical Features
    numerical_features = df.select_dtypes(include=[np.number])
    numerical_stats = numerical_features.describe().T
    print("Numerical Features Statistics:\n", numerical_stats, "\n")

    # Categorical Features
    categorical_features = df.select_dtypes(include=['object', 'category'])
    for column in categorical_features:
        print(f"Value Counts for {column}:\n{df[column].value_counts()}\n")
    return  {'numerical_stats': numerical_stats,
            'categorical_counts': {column: df[column].value_counts() for column in categorical_features}}


basicstats = generate_basic_statistics(df_traintest)

num_stats = basicstats['numerical_stats']
num_stats.to_csv('SolarPowerPrediction/rnn_article/features_stats_new.csv', index = True)
