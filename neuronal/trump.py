import json

import pandas as pd
from pathlib import Path
import numpy as np
import keras as keras
import matplotlib.pyplot as plt
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import train_test_split, GridSearchCV

path_to_data = Path('C:\git\DL4G')

users_data = json.load(open(Path(path_to_data, 'player_all_stat.json'), 'r'))
users = pd.DataFrame(users_data)
print("Before Removing users: {}".format(len(users.values)))
users = users[users['mean'] >= 80]
users = users[users['nr'] >= 100]
print("After Removing users: {}".format(len(users.values)))

rows = 28000
data = pd.read_csv(path_to_data / '2018_10_18_trump.csv', header=None, nrows=rows)
forehand = ['FH']
user = ['id']
trump = ['trump']
cards = [
    'DA', 'DK', 'DQ', 'DJ', 'D10', 'D9', 'D8', 'D7', 'D6',
    'HA', 'HK', 'HQ', 'HJ', 'H10', 'H9', 'H8', 'H7', 'H6',
    'SA', 'SK', 'SQ', 'SJ', 'S10', 'S9', 'S8', 'S7', 'S6',
    'CA', 'CK', 'CQ', 'CJ', 'C10', 'C9', 'C8', 'C7', 'C6'
]
data.columns = cards + forehand + user + trump

print("Before Removing users from data: {}".format(len(data.values)))
data = pd.merge(data, users, on='id', how='inner')
print("After Removing users: {}".format(len(data.values)))
data.drop('id', axis='columns', inplace=True)

feature_columns = forehand + cards
for color in 'DHSC':
    # Jack and nine combination
    jack_nine = '{}_J9'.format(color)
    jack_nine_combo = data['{}J'.format(color)] & data['{}9'.format(color)]
    data[jack_nine] = jack_nine_combo
    feature_columns.append(jack_nine)

    # Ace-King-Queen (Dreiblatt).
    akq = '{}_AKQ'.format(color)
    data[akq] = data['{}A'.format(color)] & data['{}K'.format(color)] & data['{}Q'.format(color)]
    feature_columns.append(akq)

    # Ace-King-nine (Dreiblatt).
    aknine = '{}_AK9'.format(color)
    data[aknine] = data['{}A'.format(color)] & data['{}K'.format(color)] & data['{}9'.format(color)]
    feature_columns.append(aknine)

    # Ace-King-Junge (Dreiblatt).
    akj = '{}_AKJ'.format(color)
    data[akj] = data['{}A'.format(color)] & data['{}K'.format(color)] & data['{}J'.format(color)]
    feature_columns.append(akj)

    # Ace-queen-Junge (Dreiblatt).
    aqj = '{}_AQJ'.format(color)
    data[aqj] = data['{}A'.format(color)] & data['{}Q'.format(color)] & data['{}J'.format(color)]
    feature_columns.append(aqj)

    # six seven eight (Dreiblatt).
    six_seven_eight = '{}_678'.format(color)
    data[six_seven_eight] = data['{}6'.format(color)] & data['{}7'.format(color)] & data['{}8'.format(color)]
    feature_columns.append(six_seven_eight)

input = data[feature_columns].values
output = keras.utils.to_categorical(data[['trump']].values)
input_size = input.shape[1]
output_size = len(np.unique(output))

input_train, input_test, output_train, output_test = (
    train_test_split(input, output, test_size=0.2, random_state=42))

def train():
    model = keras.Sequential()
    model.add(keras.layers.Dense(64, activation='relu', input_shape=[len(feature_columns)]))
    model.add(keras.layers.Dense(7, activation='softmax'))
    model.compile(loss=keras.losses.CategoricalCrossentropy(from_logits=True), optimizer='adam', metrics=['accuracy'])
    history = model.fit(input, output, validation_split=0.25, epochs=20, batch_size=64)
    print(history.history.keys())
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

def create_model(layers=1, neurons=1):
    model = keras.Sequential()
    model.add(keras.layers.Dense(neurons, input_shape=[len(feature_columns)], activation='relu'))
    for _ in range(layers - 1):
        model.add(keras.layers.Dense(neurons, activation='relu'))
    model.add(keras.layers.Dense(7, activation='softmax'))
    model.compile(loss=keras.losses.CategoricalCrossentropy(from_logits=True), optimizer='adam', metrics=['accuracy'])
    return model

def find_best_combination():
    model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=32, verbose=0)
    param_grid = {
        'layers': [1, 2],  # current: 1
        'neurons': [32, 64],  # current: 64
        'epochs': [30],  # current: 30
        'batch_size': [16]  # current: 16
    }
    grid = GridSearchCV(estimator=model, param_grid=param_grid)
    grid_result = grid.fit(input_train, output_train, validation_data=(input_test, output_test))

    print("Best: {} using {}".format(grid_result.best_score_, grid_result.best_params_))


train()
# find_best_combination()