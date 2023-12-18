import collections
import logging
import random
from pathlib import Path

import keras.models
import numpy as np
from jass.agents.agent import Agent
from jass.game.const import card_strings, PUSH, MAX_TRUMP
from jass.game.game_observation import GameObservation
from jass.game.game_util import convert_one_hot_encoded_cards_to_str_encoded_list
from jass.game.rule_schieber import RuleSchieber

playable = RuleSchieber()

path_to_data = Path('C:\git\DL4G-HS23')
class Neuronal(Agent):
    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self._rule = RuleSchieber()
        self._rng = np.random.default_rng()

    def action_play_card(self, obs: GameObservation) -> int:
        self._logger.info('Card request')
        # cards are one hot encoded
        valid_cards = self._rule.get_valid_cards_from_obs(obs)
        # convert to list and draw a value
        card = self._rng.choice(np.flatnonzero(valid_cards))
        self._logger.info('Played card: {}'.format(card_strings[card]))
        return card
    def action_trump(self, obs: GameObservation) -> int:

        model = keras.models.load_model(Path(path_to_data, 'trump.keras'))

        encoded_list = convert_one_hot_encoded_cards_to_str_encoded_list(obs.hand)
        print(encoded_list)

        finals = obs.hand

        for color in 'DHSC':
            # Jack and nine combination
            if '{}J'.format(color) in encoded_list and '{}9'.format(color) in encoded_list:
                finals = np.append(finals, 1)
            else:
                finals = np.append(finals, 0)

            if '{}A'.format(color) in encoded_list and '{}K'.format(color) in encoded_list and '{}Q'.format(color) in encoded_list:
                finals = np.append(finals, 1)
            else:
                finals = np.append(finals, 0)

            if '{}A'.format(color) in encoded_list and '{}K'.format(color) in encoded_list and '{}9'.format(color) in encoded_list:
                finals = np.append(finals, 1)
            else:
                finals = np.append(finals, 0)

            if '{}A'.format(color) in encoded_list and '{}K'.format(color) in encoded_list and '{}J'.format(color) in encoded_list:
                finals = np.append(finals, 1)
            else:
                finals = np.append(finals, 0)

            if '{}A'.format(color) in encoded_list and '{}Q'.format(color) in encoded_list and '{}J'.format(color) in encoded_list:
                finals = np.append(finals, 1)
            else:
                finals = np.append(finals, 0)

            if '{}6'.format(color) in encoded_list and '{}7'.format(color) in encoded_list and '{}8'.format(color) in encoded_list:
                finals = np.append(finals, 1)
            else:
                finals = np.append(finals, 0)

        output = model.predict(finals.reshape((1, -1)))
        return np.argmax(output)



