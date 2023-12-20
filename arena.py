# HSLU
#
# Created by Thomas Koller on 7/28/2020
#

import logging
from pathlib import Path

from jass.arena.arena import Arena

from agents.agent_rule_based_schieber import AgentRuleBasedSchieber
from neuronal.neuronal_agent import Neuronal
from neuronal.neuronal_rule_based_combined import NeuronalRuleBasedCombined


def main():
    # Set the global logging level (Set to debug or info to see more messages)
    logging.basicConfig(level=logging.ERROR)


    # setup the arena
    arena = Arena(nr_games_to_play=1000, save_filename='arena_games')
    player = AgentRuleBasedSchieber()
    my_player = NeuronalRuleBasedCombined(Path('C:\git\DL4G-HS23'))

    arena.set_players(my_player, player, my_player, player)
    print('Playing {} games'.format(arena.nr_games_to_play))
    arena.play_all_games()
    print('Average Points Team 0: {:.2f})'.format(arena.points_team_0.mean()))
    print('Average Points Team 1: {:.2f})'.format(arena.points_team_1.mean()))


if __name__ == '__main__':
    main()
