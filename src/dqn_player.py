from game import Board, Game
from dqn_gomoku_game import init_game, make_move, get_reward, draw_grid, display_grid
from dqn_utils import load_agent
import numpy as np

class DQNPlayer(object):
    def __init__(self, **kwargs):
        self.player = None
        self.width = 8
        self.filename = 'output/v_1/epoch_100/agent_2.pkl'
        self.agent = load_agent(self.filename)

    def set_player_ind(self, p: int):
        self.player = p

    def get_action(self, board: Board):
        state, available = board.current_dqn_config()
        # print('state')
        # print(state.shape)
        # print(state)
        # print('available')
        # print(available.shape)
        # print(available)
        # print(state.reshape(1, 2 * self.width**2))
        qval = self.agent.predict(state.reshape(1, 2 * self.width**2))
        # print('qval')
        # print(qval.shape)
        # print(qval)
        action = np.argmax(qval + available.reshape(1, self.width**2))
        return action 

    def __str__(self):
        return "DQNPlayer {}".format(self.player)