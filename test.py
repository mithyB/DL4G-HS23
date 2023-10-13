import numpy as np
from jass.game.game_util import *
from jass.game.game_sim import GameSim
from jass.game.game_observation import GameObservation
from jass.game.const import *
from jass.game.rule_schieber import RuleSchieber
from jass.agents.agent import Agent
from jass.agents.agent_random_schieber import AgentRandomSchieber
from jass.arena.arena import Arena

# Lets set the seed of the random number generater, so that we get the same results
np.random.seed(1)

# This distributes the cards randomly among the 4 players.
hands = deal_random_hand()
print(hands.shape)

# There is an entry for each player, to access the cards of the first player
cards = hands[0,:]
print(cards)

# This should be 9 cards
assert(cards.sum() == 9)

# The cards can be converted to other formats for easier reading or processing
print(convert_one_hot_encoded_cards_to_str_encoded_list(cards))

# Each card is encoded as a value between 0 and 35.
print(convert_one_hot_encoded_cards_to_int_encoded_list(cards))

# There is a method to count colors too
colors = count_colors(cards)
print(colors)

def havePuurWithFour(hand: np.ndarray) -> np.ndarray:
    result = np.zeros(4, dtype=int)
    # add your code here
    result[0] = hand[3]
    result[1] = hand[12]
    result[2] = hand[21]
    result[3] = hand[30]
    #
    
    return result

# assert (havePuurWithFour(cards) == [0, 0, 0, 1]).all()
assert (havePuurWithFour(cards) == [1, 0, 1, 1]).all()
cards_2 = hands[1,:]
assert (havePuurWithFour(cards_2) == [0, 0, 0, 0]).all()

# Score for each card of a color from Ace to 6

# score if the color is trump
trump_score = [15, 10, 7, 25, 6, 19, 5, 5, 5]
# score if the color is not trump
no_trump_score = [9, 7, 5, 2, 1, 0, 0, 0, 0]
# score if obenabe is selected (all colors)
obenabe_score = [14, 10, 8, 7, 5, 0, 5, 0, 0,]
# score if uneufe is selected (all colors)
uneufe_score = [0, 2, 1, 1, 5, 5, 7, 9, 11]

def calculate_trump_selection_score(cards, trump: int) -> int:
    # add your code here
    diamonds = cards[0:9]
    hearts = cards[9:18]
    spades = cards[18:27]
    clubs = cards[27:36]

    points_for_diamonds = [x * y for x, y in zip(diamonds, trump_score if trump == DIAMONDS else no_trump_score)]
    points_for_hearts = [x * y for x, y in zip(hearts, trump_score if trump == HEARTS else no_trump_score)]
    points_for_spades = [x * y for x, y in zip(spades, trump_score if trump == SPADES else no_trump_score)]
    points_for_clubs = [x * y for x, y in zip(clubs, trump_score if trump == CLUBS else no_trump_score)]

    sum_per_color = np.sum([points_for_diamonds, points_for_hearts, points_for_spades, points_for_clubs])
    
    result = np.sum(sum_per_color)

    return result

card_list = convert_one_hot_encoded_cards_to_int_encoded_list(cards)
# assert calculate_trump_selection_score(card_list, CLUBS) == 58
# assert calculate_trump_selection_score(card_list, SPADES) == 70
assert calculate_trump_selection_score(cards, CLUBS) == 58
assert calculate_trump_selection_score(cards, SPADES) == 70