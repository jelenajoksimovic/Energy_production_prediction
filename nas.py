import warnings
warnings.filterwarnings("ignore")
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import load_model
import joblib
from tensorflow import keras

import numpy as np
import tensorflow as tf
import random
import os

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

from kerastuner import HyperModel
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from keras_tuner import HyperModel, RandomSearch



print(os.getcwd())
os.chdir('C:\\Users\\Jelena\\PycharmProjects\\pythonProjects')
df_traintest = pd.read_csv('SolarPowerPrediction\\rnn_article\\traintest_features_ds.csv')

'''
#heatmap to check

df['Time'] = pd.to_datetime(df['Time'])

df['Date'] = df['Time'].dt.date
df['Hour'] = df['Time'].dt.hour

scaler = MinMaxScaler()
features = df_traintest.drop(['SolarPower'], axis=1)
features = features.drop(['Time'], axis=1)

target = df_traintest['SolarPower']

scaled_features = scaler.fit_transform(features)
scaled_target = scaler.fit_transform(target.values.reshape(-1,1))

sequence_length = 24
prediction_step = 1

# Function to create sequences
def create_sequences(features, target, sequence_length, prediction_step):
    X, y = [], []
    for i in range(len(features) - sequence_length - prediction_step + 1):
        X.append(features[i:(i + sequence_length)])
        y.append(target[i + sequence_length + prediction_step - 1])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_features, scaled_target.flatten(), sequence_length, prediction_step)

# Split the data
split_idx = int(len(X) * 0.15)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]



class LSTMHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        model = Sequential()
        model.add(LSTM(
            units=hp.Int('units', min_value=32, max_value=512, step=32),
            activation='relu',
            input_shape=self.input_shape
        ))
        model.add(Dense(1))
        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
            ),
            loss='mean_squared_error',
            metrics=['mean_squared_error']
        )
        return model

input_shape = (sequence_length, scaled_features.shape[1])

# Initialize the tuner
tuner = RandomSearch(
    LSTMHyperModel(input_shape=input_shape),
    objective='val_loss',
    max_trials=5,
    executions_per_trial=2,
    directory='tuner_results',
    project_name='lstm_tuning_v2'
)

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# Execute the search
tuner.search(X_train, y_train, epochs=30, validation_split=0.2,callbacks=[stop_early])

# Get the best model
best_model = tuner.get_best_models(num_models=1)[0]

best_hps=tuner.get_best_hyperparameters()[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")
