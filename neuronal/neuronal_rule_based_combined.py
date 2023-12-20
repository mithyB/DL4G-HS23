import logging
from pathlib import Path

import keras.models
import numpy as np
from jass.agents.agent import Agent
from jass.game.const import card_strings, OBE_ABE, UNE_UFE
from jass.game.game_observation import GameObservation
from jass.game.game_util import convert_one_hot_encoded_cards_to_str_encoded_list, count_colors
from jass.game.rule_schieber import RuleSchieber
from neuronal.neuronal_features import defined_features

class NeuronalRuleBasedCombined(Agent):

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
        # cards are one hot encoded

        if obs.trump == OBE_ABE:
            return self.play_obe_abe(obs)
        elif obs.trump == UNE_UFE:
            return self.play_one_ufe(obs)

        return self.play_with_trump(obs)

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

    def play_obe_abe(self, obs: GameObservation):
        if obs.nr_cards_in_trick == 0 or obs.nr_cards_in_trick == 2:
            # play the highest card
            return self.highest_non_zero_over_colors(obs)
        return self.lowest_non_zero_over_colors(obs)

    def play_one_ufe(self, obs: GameObservation):
        if obs.nr_cards_in_trick == 0 or obs.nr_cards_in_trick == 2:
            # play the lowest card
            return self.lowest_non_zero_over_colors(obs)
        return self.highest_non_zero_over_colors(obs)

    def play_with_trump(self, obs: GameObservation):
        playable = self._rule.get_valid_cards_from_obs(obs)
        has_trump = count_colors(playable)[obs.trump] > 0
        if obs.nr_cards_in_trick == 0:
            if has_trump:
                return self.play_highest_trump(obs)
            return self.highest_non_zero_over_colors(obs)

        if has_trump:
            return self.play_highest_trump(obs)
        return self.highest_non_zero_over_colors(obs)

    def highest_non_zero_over_colors(self, obs: GameObservation):
        cards = self._rule.get_valid_cards_from_obs(obs)
        maxValue = -1
        trump = -1
        for i in range(4):
            value = self.highest_non_zero(cards[(i * 9):((i + 1) * 9)])
            if value > maxValue:
                maxValue = value
                trump = i
        return (trump * 9) + maxValue

    def lowest_non_zero_over_colors(self, obs: GameObservation):
        cards = self._rule.get_valid_cards_from_obs(obs)
        lowestValue = 99
        trump = -1
        for i in range(4):
            value = self.lowest_non_zero(cards[(i * 9):((i + 1) * 9)])
            if value < lowestValue:
                lowestValue = value
                trump = i
        return (trump * 9) + lowestValue

    def highest_non_zero(self, param: list[int]):
        res = -1
        for i in range(len(param)):
            if param[i] == 1:
                res = i
                break
        return res

    def lowest_non_zero(self, param: list[int]):
        res = 99
        for i in range(1, len(param)+1):
            if param[-i] == 1:
                res = len(param) - i
                break
        return res

    def play_highest_trump(self, obs):
        playable = self._rule.get_valid_cards_from_obs(obs)
        if playable[9 * obs.trump + 3] == 1:
            return 9 * obs.trump + 3
        if playable[9 * obs.trump + 5] == 1:
            return 9 * obs.trump + 5
        return self.lowest_non_zero(playable[obs.trump * 9:((obs.trump + 1) * 9)]) + (obs.trump * 9)

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



