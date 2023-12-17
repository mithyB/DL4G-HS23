import json
from pathlib import Path
import os

path_to_data = Path('C:\git\DL4G')
STOP_AT_COUNT = 10_000_000
directory = os.fsencode(Path(path_to_data, 'data'))
lines = []
for file in os.listdir(directory):
    name = os.fsdecode(file)
    print("######### USING FROM {} Current Lines: {} #########".format(name, len(lines)))
    games = []
    with open(Path(path_to_data, 'data/{}'.format(name)), 'r') as file:
        for line in file:
            json_loaded = json.loads(line)
            games.append([json_loaded['game'], json_loaded['player_ids']])
            if len(lines) > STOP_AT_COUNT:
                break

    cards_lookup = [
        'DA', 'DK', 'DQ', 'DJ', 'D10', 'D9', 'D8', 'D7', 'D6',
        'HA', 'HK', 'HQ', 'HJ', 'H10', 'H9', 'H8', 'H7', 'H6',
        'SA', 'SK', 'SQ', 'SJ', 'S10', 'S9', 'S8', 'S7', 'S6',
        'CA', 'CK', 'CQ', 'CJ', 'C10', 'C9', 'C8', 'C7', 'C6'
    ]

    for playable in games:
        if len(lines) > STOP_AT_COUNT:
            break
        trump = str(playable[0]['trump'])

        for i in range(4):
            if len(lines) > STOP_AT_COUNT:
                break

            cards = []
            for trick in playable[0]['tricks']:
                index = cards_lookup.index(trick['cards'][i])
                cards.append(str(index))
            for trick in playable[0]['tricks']:
                finalized = []
                for j in range(4):
                    player_index = (j + trick['first']) % 4
                    if player_index == i:
                        break
                    index = cards_lookup.index(trick['cards'][player_index])
                    finalized.append(str(index))

                score = 0

                if i == trick['win']:
                    score = trick['points']
                elif (i + 2) % 4 == trick['win']:
                    score = trick['points']
                else:
                    score = trick['points'] * -1

                while len(finalized) < 3:
                    finalized.append(str(-1))

                index = cards_lookup.index(trick['cards'][i])
                finalized.append(str(index))
                finalized.append(','.join(cards))
                finalized.append(trump)
                finalized.append(str(score))
                finalized.append(str(playable[1][i]))
                lines.append(','.join(finalized))

with open(Path(path_to_data, 'play/final_3.csv'), 'w') as file:
    for num, line in enumerate(lines):
        if num > STOP_AT_COUNT:
            break
        file.write(f"{line}\n")
