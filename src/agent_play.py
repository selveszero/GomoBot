# -*- coding: utf-8 -*-

from __future__ import print_function
import pickle
import argparse
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from policy_value_net_numpy import PolicyValueNetNumpy
# from policy_value_net import PolicyValueNet  # Theano and Lasagne
# from policy_value_net_pytorch import PolicyValueNet  # Pytorch
# from policy_value_net_tensorflow import PolicyValueNet # Tensorflow
# from policy_value_net_keras import PolicyValueNet  # Keras
from minimax import Minimax
from dqn_player import DQNPlayer

class Human(object):
    """
    human player
    """

    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        try:
            location = input("Your move: ")
            if isinstance(location, str):  # for python3
                location = [int(n, 10) for n in location.split(",")]
            move = board.location_to_move(location)
        except Exception as e:
            move = -1
        if move == -1 or move not in board.availables:
            print("invalid move")
            move = self.get_action(board)
        return move

    def __str__(self):
        return "Human {}".format(self.player)


def run(agent1, agent2, width):
    n = 5
    height = width
    model_file = 'best_policy_8_8_5.model'
    try:
        board = Board(width=width, height=height, n_in_row=n)
        game = Game(board)

        # ############### human VS AI ###################
        def parse_agent (agent_type):
            if agent_type == 'mcts_a0':
                # load the trained policy_value_net in either Theano/Lasagne, PyTorch or TensorFlow

                # best_policy = PolicyValueNet(width, height, model_file = model_file)
                # mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=400)

                # load the provided model (trained in Theano/Lasagne) into a MCTS player written in pure numpy
                try:
                    policy_param = pickle.load(open(model_file, 'rb'))
                except:
                    policy_param = pickle.load(open(model_file, 'rb'),
                                              encoding='bytes')  # To support python3
                best_policy = PolicyValueNetNumpy(width, height, policy_param)
                player = MCTSPlayer(best_policy.policy_value_fn,
                                        c_puct=5,
                                        n_playout=400)  # set larger n_playout for better performance
            elif agent_type == 'mcts_pure':
                player = MCTS_Pure(c_puct=5, n_playout=1000)
            elif agent_type == 'minmax':
                player = Minimax()
            elif agent_type == 'dqn': 
                player = DQNPlayer()
            elif agent_type == 'human':
                player = Human()
            else:
                player = Human()
                print('Illegal Agent Type. Defaulting to human player.')
            return player
        
        player1 = parse_agent(agent1)
        player2 = parse_agent(agent2)

        # set start_player=0 for human first
        game.start_play(player1, player2, start_player=0, is_shown=1)
    except KeyboardInterrupt:
        print('\n\rquit')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent1', type=str, default='dqn')
    parser.add_argument('--agent2', type=str, default='human')
    parser.add_argument('--width', type=int, default=8)
    args = parser.parse_args()
    run(args.agent1, args.agent2, args.width)
