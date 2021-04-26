from __future__ import print_function
from game import Board, Game
import numpy as np
import copy
import random

class Minimax(object):
    def __init__(self, **kwargs):
        self.player = None
        self.max_depth = 3
        self.seq_len = 0
        self.width = 0
        self.height = 0

    def set_player_ind(self, p: int):
        self.player = p

    def get_action(self, board: Board):
        self.seq_len = board.n_in_row
        self.width = board.width
        self.height = board.height
        square_state, availables, current_player = board.current_configuration_minmax()
        minimax_output = self.MaxValue(square_state,availables, 0, current_player)
        best_action = minimax_output[1]

        return best_action

    def MaxValue(self, square_state, avaliables, depth: int, current_player: int):
        if (depth == self.max_depth):
            return [self.evaluate(square_state, current_player), 0]

        if (len(avaliables) == 0):
            return [self.evaluate(square_state, current_player), 0]

        if self.is_over(state_array, availables:
            return [self.evaluate(square_state, current_player), 0]

        maxEval = -1
        best_action = random.choice(availables)
        for place in availables:
            r,c = board.move_to_location(place)
            new_squared_state = copy.deepcopy(square_state)
            new_availables = copy.deepcopy(availables)
            new_squared_state[r,c] = current_player
            new_availables.remove(place)
            score = self.MinValue(new_squared_state, new_availables, depth + 1, 3 - current_player) # current_player has a valid value of either 1 or 2
            if score[0] > maxEval:
                maxEval = score[0]
                best_action = place

        return [maxEval, place]

    def MinValue(self, square_state, avaliables, depth: int, current_player: int):
        if (depth == self.max_depth):
            return [self.evaluate(square_state, current_player), 0]

        if (len(avaliables) == 0):
            return [self.evaluate(square_state, current_player), 0]

        if self.is_over(state_array, availables:
            return [self.evaluate(square_state, current_player), 0]

        minEval = 100000
        best_action = random.choice(availables)
        for place in availables:
            r,c = board.move_to_location(place)
            new_squared_state = copy.deepcopy(square_state)
            new_availables = copy.deepcopy(availables)
            new_squared_state[r,c] = current_player
            new_availables.remove(place)
            score = self.MaxValue(new_squared_state, new_availables, depth + 1, 3 - current_player) # current_player has a valid value of either 1 or 2
            if score[0] < minEval:
                minEval = score[0]
                best_action = place

        return [minEval, place]



    def is_over(self, state_array, avaliables):
        # Check if enough moves have been played yet
        moved = list(set(range(width * height)) - set(self.availables))
        if (len(moved) < ((self.seq_len * 2) - 1)):
            return False
        for m in moved:
            r = m // self.width
            c = m % self.width
            # # Check Horizontally
            # if ((r in range(self.width - self.seq_len + 1)) and
            #     (len(set(state_array[m:(m_self.seq_len)])) == 1)):
            #     return True
            # # Check Vertically
            # if ((c in range(self.height - self.seq_len + 1)) and
            #     (len(set(state_array[m:(m+(self.seq_len*self.width)):width])) == 1)):
            #     return True
            # Check Horizontally
            if (c <= (self.width - self.seq_len)):
                if (len(set(state_array[r,(c:(c+self.seq_len))])) == 1):
                    return True
            # Check Vertically
            if (r <= (self.height - self.self.seq_len)):
                if (len(set(state_array[(r:(r+self.seq_len)), c])) == 1):
                    return True
            # Check Diagonally
            if ((r <= (self.height - self.self.seq_len)) and (c <= (self.width - self.seq_len))):
                if (len(set(state_array[i // self.width, i % self.width] for i in range(m, m + self.seq_len * (self.width + 1), self.width + 1))) == 1):
                    return True
            # Check Anti-Diagonally
            if ((r <= (self.height - self.self.seq_len)) and (c >= (self.seq_len - 1))):
                if (len(set(state_array[i // self.width, i % self.width] for i in range(m, m + self.seq_len * (self.width - 1), self.width - 1))) == 1):
                    return True
        return False
        # # Check Horizontally
        # for r in range(self.height):
        #     for i in range(self.width - self.seq_len + 1):
        #         if ((len(set(state_array[r,(i:(i+self.seq_len))])) == 1) and (state_array[r,i] != 0)):
        #             return True
        # # Check Vertically
        # for c in range(self.width):
        #     for i in range(self.height - self.seq_len + 1):
        #         if ((len(set(state_array[(i:(i+self.seq_len)), c])) == 1) and (state_array[r,i] != 0)):
        #             return True
        # # Check Diagonally

    def evaluate(self, state_array, current_player):
        return 1
