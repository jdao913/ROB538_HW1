import random
from random import randint
import numpy as np
import math
import grid

class Qlearn:

    def __init__(self, env):
        self.env = env
        self.dim_x = self.env.dim_x
        self.dim_y = self.env.dim_y
        self.eps = .1       #Epsilon to act epsilon greedy by
        #Intialize Q table with zeros; Qtable[action][x][y]
        #Q value for state s where agent at (xa, ya) and target at (xt, yt) and action a:
        #Qtable[a][yt*dim_y + ya][xt*dim_x + xa]
        # self.Qtable = [[[0 for i in range(self.env.dim_x^2)] for i in range(self.env.dim_y^2)]
                        # for i in range(self.env.params.action_dim)]
        self.Qtable = np.zeros((self.dim_x^2, self.dim_y^2, self.env.params.action_dim))
        # Q value for state s where agent at (xa, ya) and target at (xt, yt) and action a:
        # Qtable[xt*dim_x + xa, yt*dim_y + ya, a]
        # Action 0, 1, 2, 3, correspond to up, right, down, left respectively

    #Compute epsilon greedy action according to current Q values
    def get_action(self, rover_pos, target_pos):
        if random.random > self.eps:        # Act greedily
            actionVs = self.Qtable[target_pos[0]*self.dim_x + rover_pos[0],
                                target_pos[1]*self.dim_y + rover_pos[1], :]
            direct = np.argmax(actionVs)
        else:       # Random action
            moves = [0, 1, 2, 3]
            if rover_pos[0] == 0:
                moves.remove(3)
            if rover_pos[0] == self.dim_x - 1:
                moves.remove(1)
            if rover_pos[1] == 0:
                moves.remove(2)
            if rover_pos[1] == self.dim_y - 1:
                moves.remove(0)
            direct = random.choice(moves)
        action = [-1, 0]
        if direct == 0:
            action = [0, 1]
        elif direct == 1:
            action = [1, 0]
        elif direct == 2:
            action = [0, -1]
        return action





    