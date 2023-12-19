import json

import numpy as np
import pandas as pd
from pathlib import Path
import keras as keras
import matplotlib.pyplot as plt
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import train_test_split, GridSearchCV

path_to_data = Path('C:\\Users\\micha\\code\\DL4G-HS23')

users_data = json.load(open(Path(path_to_data, 'player/player_all_stat.json'), 'r'))
users = pd.DataFrame(users_data)
# print("Before Removing users: {}".format(len(users.values)))
# users = users[users['mean'] >= 80]
# users = users[users['nr'] >= 100]
# print("After Removing users: {}".format(len(users.values)))

data = pd.read_csv(path_to_data / 'play/final_3.csv', header=None)


owned_cards = []
for i in range(9):
    owned_cards.append('owned_cards_{}'.format(i))

played_cards_column = ['played_card_0', 'played_card_1', 'played_card_2']
data.columns = played_cards_column + ['own_played_card'] + owned_cards + ['trump'] + ['points'] + ['id']

print("Before Removing users from data: {}".format(len(data.values)))
data = pd.merge(data, users, on='id', how='inner')
print("After Removing users: {}".format(len(data.values)))
data.drop('id', axis='columns', inplace=True)
data.drop('mean', axis='columns', inplace=True)
data.drop('std', axis='columns', inplace=True)
data.drop('nr', axis='columns', inplace=True)

feature_columns = played_cards_column + ['own_played_card'] + owned_cards + ['trump']
input = data[feature_columns].values
output = np.array([[1] if i > 0 else [0] for i in data[['points']].values])

input_train, input_test, output_train, output_test = (
    train_test_split(input, output, test_size=0.2, random_state=42))

print(input[0])
print(output[0])
def train():
    model = keras.Sequential()
    model.add(keras.layers.Dense(16, activation='relu', input_shape=[len(feature_columns)]))
    model.add(keras.layers.Dense(16, activation='softmax'))
    model.add(keras.layers.Dense(len(feature_columns), activation='softmax'))
    model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])
    history = model.fit(input, output, validation_split=0.2, epochs=36, batch_size=256)
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
    model.add(keras.layers.Dense(64, input_shape=[len(feature_columns)], activation='relu'))
    for _ in range(layers - 1):
        model.add(keras.layers.Dense(neurons, activation='softmax'))
    model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])
    return model


def find_best_combination():
    model = KerasClassifier(build_fn=create_model, epochs=50, batch_size=128, verbose=0)
    param_grid = {
        'layers': [1, 2, 4],
        'neurons': [16, 32, 64]
    }
    grid = GridSearchCV(estimator=model, param_grid=param_grid)
    grid_result = grid.fit(input_train, output_train, validation_data=(input_test, output_test))

    print("Best: {} using {}".format(grid_result.best_score_, grid_result.best_params_))


train()
# find_best_combination()