import logging

import numpy as np
from jass.agents.agent import Agent
from jass.game.const import PUSH, MAX_TRUMP, card_strings, OBE_ABE, UNE_UFE
from jass.game.game_observation import GameObservation
from jass.game.game_util import convert_one_hot_encoded_cards_to_str_encoded_list, count_colors
from jass.game.rule_schieber import RuleSchieber


class AgentRuleBasedSchieber(Agent):
    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self._rule = RuleSchieber()
        self._rng = np.random.default_rng()

    def action_trump(self, obs: GameObservation) -> int:

        self._logger.info('Trump request')
        self._logger.info('Cards {}'.format(', '.join(convert_one_hot_encoded_cards_to_str_encoded_list(obs.hand))))

        color = self.color_if_bube_and_9_and_more_then_2(obs)
        if color >= 0:
            self._logger.info('Playing (forehand) {} because hand has Bube and 9 and more then 2'.format(color_to_str(color)))
            return color

        color = self.color_if_bube_or_9_and_more_then_3(obs)
        if color >= 0:
            self._logger.info('Playing (forehand) {} because hand has Bube or 9 and more then 3'.format(color_to_str(color)))
            return color

        color = self.color_if_more_then_4(obs)
        if color >= 0:
            self._logger.info('Playing (forehand) {} because i have more then 5'.format(color_to_str(color)))
            return color

        color = self.obe_onde_ufe_forehand(obs)
        if color >= 0:
            self._logger.info('Playing (forehand) {} because have good enough cards'.format(color_to_str(color)))
            return color

        if obs.forehand == -1:
            self._logger.info('Pushing because my cards are bad')
            return PUSH

        color = self.obe_onde_ufe_backhand(obs)
        if color >= 0:
            self._logger.info('Playing backhand {} because have good enough cards'.format(color_to_str(color)))
            return color

        colors = count_colors(obs.hand)
        max_value_color_index = 0
        max_value = -1
        for colorIndex in range(len(colors)):
            if self.has_bube(obs, colorIndex) and self.has_9(obs, colorIndex):
                self._logger.info('Forced to take {} because i have 9 and Junge here'.format(color_to_str(color)))
                return colorIndex

            current_color_value = self.count_card_value(obs, colorIndex)
            self._logger.info('Calculated Value for color {} was {}'.format(color_to_str(colorIndex), current_color_value))
            if current_color_value > max_value:
                max_value = current_color_value
                max_value_color_index = colorIndex

        self._logger.info('Taking {} because it has a card value of {}'.format(color_to_str(max_value_color_index), max_value))
        return max_value_color_index

    def action_play_card(self, obs: GameObservation) -> int:
        self._logger.info('Card request')
        # cards are one hot encoded

        if obs.trump == OBE_ABE:
            return self.play_obe_abe(obs)
        elif obs.trump == UNE_UFE:
            return self.play_one_ufe(obs)

        return self.play_with_trump(obs)

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

    def color_if_bube_and_9_and_more_then_2(self, obs: GameObservation):
        colors = count_colors(obs.hand)
        for colorIndex in range(len(colors)):
            if colors[colorIndex] >= 3 and self.has_bube(obs, colorIndex) and self.has_9(obs, colorIndex):
                return colorIndex
        return -1

    def color_if_bube_or_9_and_more_then_3(self, obs: GameObservation):
        colors = count_colors(obs.hand)
        for colorIndex in range(len(colors)):
            if colors[colorIndex] >= 4 and (self.has_bube(obs, colorIndex) or self.has_9(obs, colorIndex)):
                return colorIndex
        return -1
    def color_if_more_then_4(self, obs: GameObservation):
        colors = count_colors(obs.hand)
        for colorIndex in range(len(colors)):
            if colors[colorIndex] >= 5:
                return colorIndex
        return -1
    def has_bube(self, obs: GameObservation, color: int):
        return obs.hand[9 * color + 3] == 1

    def has_9(self, obs: GameObservation, color: int):
        return obs.hand[9 * color + 5] == 1

    def obe_onde_ufe_forehand(self, obs: GameObservation):
        connected = self.get_connecting_one_ond_obe_abe(obs)
        if connected[0] >= 3:
            return OBE_ABE
        if connected[0] >= 2 and connected[1] >= 3:
            return OBE_ABE
        if connected[0] >= 1 and connected[1] >= 3:
            return OBE_ABE

        if connected[2] >= 3:
            return UNE_UFE
        if connected[2] >= 2 and connected[3] >= 3:
            return UNE_UFE
        if connected[2] >= 1 and connected[3] >= 3:
            return UNE_UFE
        return -1

    def obe_onde_ufe_backhand(self, obs: GameObservation):
        connected = self.get_connecting_one_ond_obe_abe(obs)
        if connected[0] >= 2:
            return OBE_ABE
        if connected[0] >= 1 and connected[1] >= 3:
            return OBE_ABE

        if connected[2] >= 2:
            return UNE_UFE
        if connected[2] >= 1 and connected[3] >= 3:
            return UNE_UFE
        return -1

    def get_connecting_one_ond_obe_abe(self, obs: GameObservation):
        connectingAssKing = 0
        assCount = 0
        connecting6and7 = 0
        sixCount = 0

        for i in range(4):
            if obs.hand[i * 9 + 0] and obs.hand[i * 9 + 1]:
                connectingAssKing = connectingAssKing + 1
            if obs.hand[i * 9]:
                assCount = assCount + 1
            if obs.hand[i * 9 + 0] and obs.hand[i * 9 + 1]:
                connectingAssKing = connectingAssKing + 1
            if obs.hand[i * 9]:
                assCount = assCount + 1

        return [connectingAssKing, assCount, connecting6and7, sixCount]

    def count_card_value(self, obs: GameObservation, color: int):
        value = 0
        # Junge is worth 9, 9 is worth 8, Ass 7, König 6, etc.
        value_of_index = [7, 6, 9, 5, 4, 6, 3, 2, 1]
        for i in range(len(value_of_index)):
            if obs.hand[9 * color + i] == 1:
                value = value + value_of_index[i]
        return value

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


def color_to_str(color: int):
    return ["Diamond", "Heart", "Spades", "Clubs", "OBE_ABE", "UNE_UFE"][color]