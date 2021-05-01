from __future__ import print_function
from game import Board, Game
import numpy as np
import copy
import random

class Minimax(object):
    def __init__(self, **kwargs):
        self.player = None
        self.max_depth = 2
        self.seq_len = 0
        self.width = 1
        self.height = 1

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

    def move_to_location(self, move):
        h = move // self.width
        w = move % self.width
        return [h, w]

    def MaxValue(self, square_state, availables, depth: int, current_player: int):
        if (depth == self.max_depth):
            return [self.evaluate(square_state, current_player), 0]

        if (len(availables) == 0):
            return [self.evaluate(square_state, current_player), 0]

        game_over = self.is_over(square_state, availables)
        if game_over:
            if (current_player == self.player):
                return [-100000, 0]
            else:
                return [100000, 0]
            # return [self.evaluate(square_state, current_player), 0]

        maxEval = -100001
        best_action = random.choice(availables)
        for place in availables:
            r,c = self.move_to_location(place)
            new_squared_state = copy.deepcopy(square_state)
            new_availables = copy.deepcopy(availables)
            new_squared_state[r,c] = current_player
            new_availables.remove(place)
            score = self.MinValue(new_squared_state, new_availables, depth + 1, 3 - current_player) # current_player has a valid value of either 1 or 2
            print("Min Value at location ", self.move_to_location(place), " is ", score)
            if score[0] > maxEval:
                maxEval = score[0]
                best_action = place
        print("My maxEval is ", maxEval, " and place is ", best_action, " and availables is ", availables)
        return [maxEval, best_action]

    def MinValue(self, square_state, availables, depth: int, current_player: int):
        if (depth == self.max_depth):
            return [self.evaluate(square_state, current_player), 0]

        if (len(availables) == 0):
            return [self.evaluate(square_state, current_player), 0]

        game_over = self.is_over(square_state, availables)
        if game_over:
                if (current_player == self.player):
                    return [-100000, 0]
                else:
                    return [100000, 0]

        minEval = 100001
        best_action = random.choice(availables)
        for place in availables:
            r,c = self.move_to_location(place)
            new_squared_state = copy.deepcopy(square_state)
            new_availables = copy.deepcopy(availables)
            new_squared_state[r,c] = current_player
            new_availables.remove(place)
            score = self.MaxValue(new_squared_state, new_availables, depth + 1, 3 - current_player) # current_player has a valid value of either 1 or 2
            if score[0] < minEval:
                minEval = score[0]
                best_action = place

        return [minEval, best_action]



    def is_over(self, state_array, availables):
        # Check if enough moves have been played yet
        moved = list(set(range(self.width * self.height)) - set(availables))
        if (len(moved) < ((self.seq_len * 2) - 1)):
            return False
        for m in moved:
            r = m // self.width
            c = m % self.width

            # Check Horizontally
            if (c <= (self.width - self.seq_len)):
                if (len(set(state_array[r,c:(c+self.seq_len)])) == 1):
                    return True
            # Check Vertically
            if (r <= (self.height - self.seq_len)):
                if (len(set(state_array[r:(r+self.seq_len), c])) == 1):
                    return True
            # Check Diagonally
            if ((r <= (self.height - self.seq_len)) and (c <= (self.width - self.seq_len))):
                if (len(set(state_array[i // self.width, i % self.width] for i in range(m, m + self.seq_len * (self.width + 1), self.width + 1))) == 1):
                    return True
            # Check Anti-Diagonally
            if ((r <= (self.height - self.seq_len)) and (c >= (self.seq_len - 1))):
                if (len(set(state_array[i // self.width, i % self.width] for i in range(m, m + self.seq_len * (self.width - 1), self.width - 1))) == 1):
                    return True
        return False


    def evaluate(self, state_array, current_player):
        utility = 0
        # Evaluate Horizontally
        for r in range(self.height):
            for c in range(self.width - self.seq_len + 1):
                good_count = 0
                bad_count = 0
                for i in range(self.seq_len):
                    if (state_array[r,c+i] == current_player):
                        good_count += 1
                    elif (state_array[r,c+i] == (3-current_player)):
                        bad_count += 1
                if (good_count > 0):
                    if (bad_count == 0):
                        utility += 10**good_count
                else:
                    if (bad_count > 0):
                        utility -= 10**bad_count
        # Evaluate Vertically
        for c in range(self.width):
            for r in range(self.height - self.seq_len + 1):
                good_count = 0
                bad_count = 0
                for i in range(self.seq_len):
                    if (state_array[r+i,c] == current_player):
                        good_count += 1
                    elif (state_array[r+i,c] == (3-current_player)):
                        bad_count += 1
                if (good_count > 0):
                    if (bad_count == 0):
                        utility += 10**good_count
                else:
                    if (bad_count > 0):
                        utility -= 10**bad_count
        # Evaluate Diagonally
        for r in range(self.height - self.seq_len + 1):
            for c in range(self.width - self.seq_len + 1):
                good_count = 0
                bad_count = 0
                for i in range(self.seq_len):
                    if (state_array[r+i,c+i] == current_player):
                        good_count += 1
                    elif (state_array[r+i,c+i] == (3-current_player)):
                        bad_count += 1
                if (good_count > 0):
                    if (bad_count == 0):
                        utility += 10**good_count
                else:
                    if (bad_count > 0):
                        utility -= 10**bad_count
        # Evaluate Anti-Diagonally
        for r in range(self.height - self.seq_len + 1):
            for c in range(self.seq_len,self.width):
                good_count = 0
                bad_count = 0
                for i in range(self.seq_len):
                    if (state_array[r+i,c-i] == current_player):
                        good_count += 1
                    elif (state_array[r+i,c-i] == (3-current_player)):
                        bad_count += 1
                if (good_count > 0):
                    if (bad_count == 0):
                        utility += 10**good_count
                else:
                    if (bad_count > 0):
                        utility -= 10**bad_count

        return utility

    def __str__(self):
        return "Minimax {}".format(self.player)
