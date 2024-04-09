#Features

import warnings

warnings.filterwarnings("ignore")
import random

random.seed(42)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import datetime
from tensorflow.keras.models import load_model
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import os

print(os.getcwd())
os.chdir('C:\\Users\\Jelena\\PycharmProjects\\pythonProjects')



# Feature Engineering
def feature_processing():
    df = pd.read_csv('SolarPowerPrediction\\rnn_article\\traintest_data.csv')
    df['Time'] = pd.to_datetime(df['Time'])
    df.set_index('Time', inplace=True)
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    df['Day'] = df.index.day
    df['Day_of_year'] = df.index.dayofyear

    #apply sin and cos transformations to encode cyclical patterns at different time scales, including daily, monthly, and yearly levels
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)

    df['sin_day_of_month'] = np.sin(2 * np.pi * df['Day'] / 30) # Assuming 30 as a rough average of days in a month
    df['cos_day_of_month'] = np.cos(2 * np.pi * df['Day'] / 30)

    #df['sin_day_of_year'] = np.sin(2 * np.pi * df['Day_of_year'] / 365)
    #df['cos_day_of_year'] = np.cos(2 * np.pi * df['Day_of_year'] / 365)

    # Calculate daylight duration in hours
    df['sunrise_dt'] = pd.to_datetime(df['SunriseDT'])
    df['sunset_dt'] = pd.to_datetime(df['SunsetDT'])
    df['daylight_duration'] = (df['sunset_dt'] - df['sunrise_dt']).dt.total_seconds() / 3600

    # Binary feature for daylight availability
    #df['is_daylight'] = ((df.index >= df['sunrise_dt']) & (df.index <= df['sunset_dt'])).astype(int)

    # Categorical to numerical values

    # Initialize the label encoder
    label_encoder_weather = LabelEncoder()
    label_encoder_description = LabelEncoder()

    # Apply Label Encoding to each column
    df['Weather_encoded'] = label_encoder_weather.fit_transform(df['Weather'])
    df['WeatherDescription_encoded'] = label_encoder_description.fit_transform(df['WeatherDescription'])
    joblib.dump(label_encoder_weather, 'label_encoder_weather.joblib')
    joblib.dump(label_encoder_description, 'label_encoder_description.joblib')


    # Select features for modeling
    selected_features = ['Temperature', 'Clouds', 'Visibility', 'Humidity', 'Pressure', 'WindSpeed', 'hour_sin',
                         'hour_cos', 'sin_day_of_month', 'cos_day_of_month','daylight_duration','Weather_encoded'
                         ,'WeatherDescription_encoded']

    df_selected = df[selected_features]
    df_selected['SolarPower'] = df['SolarPower']*10  # Target variable to be for whole RCNM
    df_selected.to_csv('SolarPowerPrediction/rnn_article/traintest_features_ds.csv', index=True)

    return df, df_selected

#Transform Serving Dataset for predictions

#df_history = pd.read_csv('SolarPowerPrediction/HistoryData.csv')

def serving_data(df):
    df_serving = pd.read_csv('SolarPowerPrediction\\rnn_article\\serving_data.csv')
    # Assuming df_weather is your DataFrame containing the WeatherForecast table
    #'Time' column to datetime format
    df_serving['Time'] = pd.to_datetime(df_serving['Time'])
    # Set 'Time' as the index
    df_serving.set_index('Time', inplace=True)

    #1: Add columns to align with HistoryData table
    # Convert strings to datetime
    df['SunriseDT'] = pd.to_datetime(df['SunriseDT'])
    df['SunsetDT'] = pd.to_datetime(df['SunsetDT'])
    # Extract the last known sunrise and sunset times
    last_sunrise_time = df.iloc[-1]['SunriseDT'].time()
    last_sunset_time = df.iloc[-1]['SunsetDT'].time()
    # Assuming WeatherForecast is loaded into df_weather as before
    sunrise_time_str = last_sunrise_time.strftime('%H:%M:%S')
    sunset_time_str = last_sunset_time.strftime('%H:%M:%S')
    df_serving['date'] = df_serving.index.date

    # Construct sunrise and sunset datetime objects
    df_serving['sunrise_dt'] = df_serving['date'].apply(lambda x: pd.to_datetime(str(x) + ' ' + sunrise_time_str))
    df_serving['sunset_dt'] = df_serving['date'].apply(lambda x: pd.to_datetime(str(x) + ' ' + sunset_time_str))

    # Drop the temporary 'date' column if no longer needed
    df_serving.drop(columns=['date'], inplace=True)

    df_serving['daylight_duration'] = (df_serving['sunset_dt'] - df_serving['sunrise_dt']).dt.total_seconds() / 3600

    # Binary feature for daylight availability
    #df_serving['is_daylight'] = ((df_serving.index >= df_serving['sunrise_dt']) & (df_serving.index <= df_serving['sunset_dt'])).astype(int)

    # Add cyclic features for time encoding

    df_serving['hour_sin'] = np.sin(2 * np.pi * df_serving.index.hour / 24)
    df_serving['hour_cos'] = np.cos(2 * np.pi * df_serving.index.hour / 24)

    df_serving['sin_day_of_month'] = np.sin(2 * np.pi * df_serving.index.day / 30)  # Assuming 30 as a rough average of days in a month
    df_serving['cos_day_of_month'] = np.cos(2 * np.pi * df_serving.index.day / 30)

    #df_serving['sin_day_of_year'] = np.sin(2 * np.pi * df_serving.index.dayofyear / 365)
    #df_serving['cos_day_of_year'] = np.cos(2 * np.pi * df_serving.index.dayofyear / 365)

    label_encoder_weather = joblib.load('label_encoder_weather.joblib')
    label_encoder_description = joblib.load('label_encoder_description.joblib')

    df_serving['Weather_encoded'] = label_encoder_weather.transform(df_serving['Weather'])
    df_serving['WeatherDescription_encoded'] = label_encoder_description.transform(df_serving['WeatherDescription'])


    #2: Drop unnecessary columns that are not in training dataset (df_selected)
    #df_weather.columns
    #df_train.columns
    df_serving.drop('Id', axis=1, inplace=True)
    df_serving.drop('sunset_dt', axis=1, inplace=True)
    df_serving.drop('sunrise_dt', axis=1, inplace=True)
    df_serving.drop('WeatherDescription', axis=1, inplace=True)
    df_serving.drop('Weather', axis=1, inplace=True)
    df_serving.drop('Rain', axis=1, inplace=True)

    #3: adjust the column order to be aligned with training dataset
    desired_order = ['Temperature', 'Clouds', 'Visibility', 'Humidity', 'Pressure', 'WindSpeed', 'hour_sin', 'hour_cos',
                     'sin_day_of_month','cos_day_of_month', 'daylight_duration',
                     'Weather_encoded','WeatherDescription_encoded']

    df_serving_reordered = df_serving[desired_order]
    print(df_serving_reordered.columns)

    #4: Expand the df_weather dataset with 15 min intervals
    # Generate a new DataFrame with 15-minute intervals
    new_index = pd.date_range(start=df_serving_reordered.index.min(), end=df_serving_reordered.index.max(), freq='15T')
    df_weather_expanded = df_serving_reordered.reindex(new_index)

    # Forward fill the NaN values
    df_weather_expanded.ffill(inplace=True)
    print(df_weather_expanded)
    df_weather_expanded.to_csv('SolarPowerPrediction/rnn_article/serving_features_ds.csv', index=True)



if __name__ == "__main__":
    df, df_selected = feature_processing()
    serving_data(df)
    print(df_selected)
    '''
    X = df_selected.drop('SolarPower', axis=1)  # Features
    y = df_selected['SolarPower']  # Target variable

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    # Get feature importance
    feature_importance = model.feature_importances_
    importances = pd.Series(feature_importance, index=X_train.columns).sort_values(ascending=False)

    plt.figure(figsize=(6,8))
    sns.barplot(x=importances, y=importances.index)

    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.title('Feature Importance')
    plt.tight_layout
    plt.show()
    '''
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Assuming df is your DataFrame with features and the target variable 'SolarPower'
    corr_matrix = df_selected.corr()  # Calculate correlation matrix

    # Extract correlation with the target variable
    target_correlation = corr_matrix['SolarPower'].sort_values(ascending=False)

    # Generate a mask for the upper triangle
    #mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap =sns.color_palette("coolwarm", as_cmap=True)


    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix.abs(), annot=True, fmt=".2f", cmap=cmap, center=0,
                linewidths=.5, cbar_kws={"shrink": .5})

    plt.xticks(rotation=45, ha='right')  # Rotate the x-axis labels for better readability
    plt.yticks(rotation=0)  # Keep the y-axis labels horizontal
    plt.title('Feature Correlation Matrix (Absolute Values)')

    plt.tight_layout()  # Adjust the layout to make room for the tick labels
    plt.savefig('SolarPowerPrediction/rnn_article/feature_corr_matrix.png')

    plt.show()

    '''
    # Plotting
    plt.figure(figsize=(10, 8))
    sns.barplot(x=target_correlation.index, y=target_correlation.values)
    plt.xticks(rotation=45)  # Rotate labels for better readability
    plt.ylabel('Pearson Correlation Coefficient')
    plt.title('Feature Importance based on Pearson Correlation with SolarPower')

    plt.show()
    '''
