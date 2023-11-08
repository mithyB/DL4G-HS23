import random
import logging
import numpy as np
from jass.agents.agent import Agent
from jass.game.rule_schieber import RuleSchieber
from jass.game.const import HEARTS, DIAMONDS, CLUBS, SPADES, OBE_ABE, UNE_UFE
from jass.game.game_observation import GameObservation
from jass.game.game_util import convert_one_hot_encoded_cards_to_str_encoded_list

class AgentCarlo(Agent):

    trump_score = [15, 10, 7, 25, 6, 19, 5, 5, 5]
    no_trump_score = [9, 7, 5, 2, 1, 0, 0, 0, 0]
    obenabe_score = [14, 10, 8, 7, 5, 0, 5, 0, 0,]
    uneufe_score = [0, 2, 1, 1, 5, 5, 7, 9, 11]
    
    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self._rule = RuleSchieber()

    def action_trump(self, obs: GameObservation) -> int:

        self._logger.info('Trump request')
        self._logger.info('Cards {}'.format(', '.join(convert_one_hot_encoded_cards_to_str_encoded_list(obs.hand))))

        diamonds = obs.hand[0:9]
        hearts = obs.hand[9:18]
        spades = obs.hand[18:27]
        clubs = obs.hand[27:36]

        trumps = [DIAMONDS, HEARTS, SPADES, CLUBS, OBE_ABE, UNE_UFE]
        result_for_trump = []

        for trump in trumps:
            points_for_diamonds = [x * y for x, y in zip(diamonds, self.trump_score if trump == DIAMONDS else self.uneufe_score if trump == UNE_UFE else self.obenabe_score if trump == OBE_ABE else self.no_trump_score)]
            points_for_hearts = [x * y for x, y in zip(hearts, self.trump_score if trump == HEARTS else self.uneufe_score if trump == UNE_UFE else self.obenabe_score if trump == OBE_ABE else self.no_trump_score)]
            points_for_spades = [x * y for x, y in zip(spades, self.trump_score if trump == SPADES else self.uneufe_score if trump == UNE_UFE else self.obenabe_score if trump == OBE_ABE else self.no_trump_score)]
            points_for_clubs = [x * y for x, y in zip(clubs, self.trump_score if trump == CLUBS else self.uneufe_score if trump == UNE_UFE else self.obenabe_score if trump == OBE_ABE else self.no_trump_score)]

            sum_per_color = np.sum([points_for_diamonds, points_for_hearts, points_for_spades, points_for_clubs])
            result = np.sum(sum_per_color)

            result_for_trump.append(result)

        trump = trumps[np.argmax(result_for_trump)]

        self._logger.info('Trump {}'.format(trump))

        return trump

    def action_play_card(self, obs: GameObservation) -> int:

        self._logger.info('Card request')
        self._logger.info('Cards {}'.format(', '.join(convert_one_hot_encoded_cards_to_str_encoded_list(obs.hand))))

        cards = self._rule.get_valid_cards_from_obs(obs)

        if obs.trump == DIAMONDS:
            score_template = [item for sublist in [self.trump_score, self.no_trump_score, self.no_trump_score, self.no_trump_score] for item in sublist]

        if obs.trump == HEARTS:
            score_template = [item for sublist in [self.no_trump_score, self.trump_score, self.no_trump_score, self.no_trump_score] for item in sublist]

        if obs.trump == SPADES:
            score_template = [item for sublist in [self.no_trump_score, self.no_trump_score, self.trump_score, self.no_trump_score] for item in sublist]

        if obs.trump == CLUBS:
            score_template = [item for sublist in [self.no_trump_score, self.no_trump_score, self.no_trump_score, self.trump_score] for item in sublist]

        if obs.trump == OBE_ABE:
            score_template = [item for sublist in [self.obenabe_score, self.obenabe_score, self.obenabe_score, self.obenabe_score] for item in sublist]

        if obs.trump == UNE_UFE:
            score_template = [item for sublist in [self.uneufe_score, self.uneufe_score, self.uneufe_score, self.uneufe_score] for item in sublist]

        score_for_all_cards = [(x+1) * y for x, y in zip(score_template, cards)]
        card = np.argmax(score_for_all_cards)

        self._logger.info('Card {}'.format(card))

        return card

        # if obs.trump == DIAMONDS:
        #     # [ 0, 0,  1,  0, 0,  0, 1, 1, 0]
        #     # [15, 10, 7, 25, 6, 19, 5, 5, 5]
        #     # [ 0, 0,  7,  0, 0,  0, 5, 5, 0]
        #     # index = 3
        #     score_for_all_cards = [...trump_score, ...no_trump_score, ...no_trump_score, ...no_trump_score]


        #     sorted_cards_by_score = [[x for _, x in sorted(zip(self.trump_score, diamonds), reverse=True)]]
        #     sorted_cards_by_score.append([x for _, x in sorted(zip(self.no_trump_score, hearts), reverse=True)])
        #     sorted_cards_by_score.append([x for _, x in sorted(zip(self.no_trump_score, spades), reverse=True)])
        #     sorted_cards_by_score.append([x for _, x in sorted(zip(self.no_trump_score, clubs), reverse=True)])

        # if obs.trump == HEARTS:
        #     sorted_cards_by_score = [[x for _, x in sorted(zip(self.trump_score, hearts), reverse=True)]]
        #     sorted_cards_by_score.append([x for _, x in sorted(zip(self.no_trump_score, diamonds), reverse=True)])
        #     sorted_cards_by_score.append([x for _, x in sorted(zip(self.no_trump_score, spades), reverse=True)])
        #     sorted_cards_by_score.append([x for _, x in sorted(zip(self.no_trump_score, clubs), reverse=True)])

        #     sorted_cards_by_score = [item for sublist in sorted_cards_by_score for item in sublist]
        #     card = next((i for i, x in enumerate(sorted_cards_by_score) if x), None)
            
        #     if card >= 0 and card < 9:
        #         card += 9
        #     if card >= 9 and card < 18:
        #         card -= 9
        #     return card

        # if obs.trump == SPADES:
        #     sorted_cards_by_score = [[x for _, x in sorted(zip(self.trump_score, spades), reverse=True)]]
        #     sorted_cards_by_score.append([x for _, x in sorted(zip(self.no_trump_score, diamonds), reverse=True)])
        #     sorted_cards_by_score.append([x for _, x in sorted(zip(self.no_trump_score, hearts), reverse=True)])
        #     sorted_cards_by_score.append([x for _, x in sorted(zip(self.no_trump_score, clubs), reverse=True)])
            
        #     sorted_cards_by_score = [item for sublist in sorted_cards_by_score for item in sublist]
        #     card = next((i for i, x in enumerate(sorted_cards_by_score) if x), None)
            
        #     if card >= 0 and card < 9:
        #         card += 18
        #     if card >= 9 and card < 27:
        #         card -= 9
        #     return card

        # if obs.trump == CLUBS:
        #     sorted_cards_by_score = [[x for _, x in sorted(zip(self.trump_score, clubs), reverse=True)]]
        #     sorted_cards_by_score.append([x for _, x in sorted(zip(self.no_trump_score, diamonds), reverse=True)])
        #     sorted_cards_by_score.append([x for _, x in sorted(zip(self.no_trump_score, hearts), reverse=True)])
        #     sorted_cards_by_score.append([x for _, x in sorted(zip(self.no_trump_score, spades), reverse=True)])
            
        #     sorted_cards_by_score = [item for sublist in sorted_cards_by_score for item in sublist]
        #     card = next((i for i, x in enumerate(sorted_cards_by_score) if x), None)
            
        #     if card >= 0 and card < 9:
        #         card += 27
        #     if card >= 9 and card < 36:
        #         card -= 9
        #     return card

        # if obs.trump == OBE_ABE:
        #     for i in range(9):
        #         if cards[i] == 1:
        #             return card
        #         if cards[i+9] == 1:
        #             return card
        #         if cards[i+18] == 1:
        #             return card
        #         if cards[i+27] == 1:
        #             return card
                
        # for i in range(9):
        #     if cards[9-i] == 1:
        #         return card
        #     if cards[18-i] == 1:
        #         return card
        #     if cards[27-i] == 1:
        #         return card
        #     if cards[36-i] == 1:
        #         return card

        # self._logger.info('Card {}'.format(card))

        # return card