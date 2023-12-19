import json

import pandas as pd
from pathlib import Path
import keras as keras
import matplotlib.pyplot as plt
from neuronal.neuronal_features import defined_features


class TrumpTrain:

    cards = [
        'DA', 'DK', 'DQ', 'DJ', 'D10', 'D9', 'D8', 'D7', 'D6',
        'HA', 'HK', 'HQ', 'HJ', 'H10', 'H9', 'H8', 'H7', 'H6',
        'SA', 'SK', 'SQ', 'SJ', 'S10', 'S9', 'S8', 'S7', 'S6',
        'CA', 'CK', 'CQ', 'CJ', 'C10', 'C9', 'C8', 'C7', 'C6'
    ]

    path_to_data = Path('C:\git\DL4G')
    player_file = 'player_all_stat.json'
    trump_file = '2018_10_18_trump.csv'
    save_model_path = 'trump.keras'
    rows = 280_000
    data = None
    feature_columns = []

    def __init__(self):
        users = self.load_and_filter_users()
        self.data = self.load_trump_data()
        self.remove_users(users)
        self.feature_columns = self.create_feature_columns()

        input_layers = self.data[self.feature_columns].values
        self.debug_input_layers(input_layers)
        output = keras.utils.to_categorical(self.data[['trump']].values)

        self.train(input_layers, output)

    def load_and_filter_users(self):
        # load and filter users
        users_data = json.load(open(Path(self.path_to_data, self.player_file), 'r'))
        users = pd.DataFrame(users_data)
        print("Before Removing users: {}".format(len(users.values)))
        users = users[users['mean'] >= 80]
        users = users[users['nr'] >= 100]
        print("After Removing users: {}".format(len(users.values)))
        return users

    def load_trump_data(self):
        # load trump file
        data = pd.read_csv(self.path_to_data / self.trump_file, header=None, nrows=self.rows)
        # define trump columns
        data.columns = self.cards + ['FH'] + ['id'] + ['trump']
        return data

    def remove_users(self, users):
        # remove bas users from data
        print("Before Removing users from data: {}".format(len(self.data.values)))
        data = pd.merge(self.data, users, on='id', how='inner')
        print("After Removing users: {}".format(len(data.values)))
        data.drop('id', axis='columns', inplace=True)
        return data

    def create_feature_columns(self):
        # define feature columns (add own features)
        feature_columns = ['FH'] + self.cards
        for color in 'DHSC':
            for feature in defined_features:
                feature_classified_name = '{}_{}'.format(color, feature)
                feature_columns.append(feature_classified_name)
                if len(feature) == 2:
                    self.data[feature_classified_name] = (
                        self.data['{}{}'.format(color, feature[0])] &
                        self.data['{}{}'.format(color, feature[1])])
                elif len(feature) == 3:
                    self.data[feature_classified_name] = (
                        self.data['{}{}'.format(color, feature[0])] &
                        self.data['{}{}'.format(color, feature[1])] &
                        self.data['{}{}'.format(color, feature[2])])
                else:
                    raise Exception('Only features of length 2 or 3 allowed')
        return feature_columns

    def train(self, input_layers, output_layers):
        model = keras.Sequential()
        model.add(keras.layers.Dense(64, activation='relu', input_shape=[len(self.feature_columns)]))
        model.add(keras.layers.Dense(7, activation='softmax'))
        model.compile(loss=keras.losses.CategoricalCrossentropy(from_logits=True), optimizer='adam', metrics=['accuracy'])
        history = model.fit(input_layers, output_layers, validation_split=0.25, epochs=20, batch_size=64)

        model.save(Path(self.path_to_data, self.save_model_path))

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

    def debug_input_layers(self, input_layer):
        for i in range(1):
            line_1 = []
            line_2 = []
            max_length = 0
            for col_name in self.feature_columns:
                if len(col_name) > max_length:
                    max_length = len(col_name)
            for num, column in enumerate(input_layer[i]):
                line_1_appender = self.feature_columns[num]
                line_2_appender = str(column)
                while len(line_1_appender) < max_length:
                    line_1_appender = line_1_appender + ' '
                while len(line_2_appender) < max_length:
                    line_2_appender = line_2_appender + ' '
                line_1.append(line_1_appender)
                line_2.append(line_2_appender)
            print('#### Debug Element {} ####'.format(i))
            print(' '.join(line_1))
            print(' '.join(line_2))

    """
    #TODO if needed, make this class bases to work again \._./
    
    ## find_best_combination()
    # choose either `train` or `find_best_combination`:
    # => find_best_combination: tries to find the best combination between layers, epochs, etc.
    
    def create_model(self, layers=1, neurons=1, feat_columns):
        model = keras.Sequential()
        model.add(keras.layers.Dense(neurons, input_shape=[len(self.feature_columns)], activation='relu'))
        for _ in range(layers - 1):
            model.add(keras.layers.Dense(neurons, activation='relu'))
        model.add(keras.layers.Dense(7, activation='softmax'))
        model.compile(loss=keras.losses.CategoricalCrossentropy(from_logits=True), optimizer='adam', metrics=['accuracy'])
        return model

    def find_best_combination():
        model = KerasClassifier(build_fn=self.create_model, epochs=10, batch_size=32, verbose=0)
        param_grid = {
            'layers': [1, 2],  # current: 1
            'neurons': [32, 64],  # current: 64
            'epochs': [30],  # current: 30
            'batch_size': [16]  # current: 16
        }
        grid = GridSearchCV(estimator=model, param_grid=param_grid)
        grid_result = grid.fit(input_train, output_train, validation_data=(input_test, output_test))
        print("Best: {} using {}".format(grid_result.best_score_, grid_result.best_params_))
    """


if __name__ == "__main__":
    # whatever, a clean method approach would have had the same result.
    TrumpTrain()
