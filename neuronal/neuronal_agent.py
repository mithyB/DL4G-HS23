import logging
from pathlib import Path

import keras.models
import numpy as np
from jass.agents.agent import Agent
from jass.game.const import card_strings
from jass.game.game_observation import GameObservation
from jass.game.game_util import convert_one_hot_encoded_cards_to_str_encoded_list
from jass.game.rule_schieber import RuleSchieber
from neuronal.neuronal_features import defined_features

class Neuronal(Agent):

    play_model = None
    trump_model = None
    def __init__(self, path):
        self._logger = logging.getLogger(__name__)
        self._rule = RuleSchieber()
        self._rng = np.random.default_rng()
        self.path_to_data = path
        self.play_model = keras.models.load_model(Path(self.path_to_data, 'data.keras'))
        self.trump_model = keras.models.load_model(Path(self.path_to_data, 'trump.keras'))

    def action_play_card(self, obs: GameObservation) -> int:
        self._logger.info('Card request')

        # input should be:
        # 0 - 2: [0-35] for each player
        # 3: own card played [0-36]: get best for playable card for this?
        # 4 - 13: owned cards: [0-8] all?
        # 14: trump
        input = np.zeros(shape=[14], dtype=np.int32)
        input[13] = obs.trump

        for num, current_trick in enumerate(obs.current_trick):
            input[num] = current_trick

        j = 0
        for num, card in enumerate(obs.hand):
            if card != 0:
                input[j + 4] = num
                j = j + 1

        valid_cards = self._rule.get_valid_cards_from_obs(obs)
        results = []
        for num, v_c in enumerate(valid_cards):
            if not v_c == 0:
                input[3] = num
                results.append([num, np.max(self.play_model.predict(input.reshape((1, -1))))])


        max = -1
        res = -1
        for r in results:
            if r[1] > max:
                max = r[1]
                res = r[0]

        if not res == -1:
            self._logger.info('Playing from neuronal {}'.format(res))
            return res

        # cards are one hot encoded
        valid_cards = self._rule.get_valid_cards_from_obs(obs)
        # convert to list and draw a value
        card = self._rng.choice(np.flatnonzero(valid_cards))
        self._logger.info('Played card: {}'.format(card_strings[card]))
        return card

    def action_trump(self, obs: GameObservation) -> int:
        encoded_list = convert_one_hot_encoded_cards_to_str_encoded_list(obs.hand)
        finals = obs.hand

        # Add features
        for color in 'DHSC':
            for feature in defined_features:
                has_all = True
                for feature_card in feature:
                    if not '{}{}'.format(color, feature_card) in encoded_list:
                        has_all = False
                        break
                if has_all:
                    finals = np.append(finals, 1)
                else:
                    finals = np.append(finals, 0)

        self.debug(finals)

        # reshape because it needs to be a 2-Dimensional Array
        output = self.trump_model.predict(finals.reshape((1, -1)))
        return np.argmax(output)

    def debug(self, finals):
        if not self._logger.isEnabledFor(10):
            return
        line_1 = []
        line_2 = []
        max_length = 0
        for col_name in defined_features:
            if len(col_name) > max_length:
                max_length = len(col_name)
        line_ones = []
        for card in card_strings:
            line_ones.append(card)
        for color in 'DHSC':
            for def_feature in defined_features:
                line_ones.append("{}_{}".format(color, def_feature))
        for num, final in enumerate(finals):
            line_2_appender = str(final)
            line_1_appender = line_ones[num]
            while len(line_1_appender) < max_length:
                line_1_appender = line_1_appender + ' '
            while len(line_2_appender) < max_length:
                line_2_appender = line_2_appender + ' '

            line_1.append(line_1_appender)
            line_2.append(line_2_appender)
        self._logger.debug('#### Debug Element Play ####')
        self._logger.debug(' '.join(line_1))
        self._logger.debug(' '.join(line_2))



