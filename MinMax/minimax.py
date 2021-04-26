from __future__ import print_function
from game import Board, Game
import numpy as np
import copy
import random

class Minimax(object):
    def __init__(self, **kwargs):
        self.player = None
        self.max_depth = 3

    def set_player_ind(self, p: int):
        self.player = p

    def get_action(self, board: Board):
        square_state, availables, current_player = board.current_configuration_minmax()
        best_action = random.choice(availables)
        best_score = -1

        for place in availables:
            r,c = board.move_to_location(place)
            new_squared_state = copy.deepcopy(square_state)
            new_availables = copy.deepcopy(availables)
            new_squared_state[r,c] = current_player
            new_availables.remove(place)
            
