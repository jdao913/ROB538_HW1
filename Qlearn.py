import random
from random import randint
import numpy as np
import math
from grid import * 
import time
# import matplotlib.pyplot as plt

class Qlearn:

    def __init__(self, env):
        self.env = env
        self.dim_x = self.env.dim_x
        self.dim_y = self.env.dim_y
        self.eps = 0        #Epsilon to act epsilon greedy by
        self.alpha = .1     #Learning rate
        self.gamma = .8     #Discount factor
        # Intialize Q table with zeros; Qtable[action][x][y]
        # Q value for state s where agent at (xa, ya) and target at (xt, yt) and action a:
        # Qtable[xt*dim_x + xa, yt*dim_y + ya, a]
        # Action 0, 1, 2, 3, correspond to up, right, down, left respectively
        self.Qtable = np.zeros((self.dim_x**2, self.dim_y**2, self.env.params.action_dim))

    #Compute epsilon greedy action according to current Q values
    def get_action(self, rover_pos, target_pos):
        if random.random() > self.eps:        # Act greedily
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
        return direct, action
    
    def train(self, niter):
        mem_buff = np.zeros(100)
        mem_iter = 0
        filename = "Qtest.npy"
        for i in range(niter):
            self.env.reset()
            done = False
            while not done:
                rov_old = self.env.rover_pos
                tar_old = self.env.target_pos
                direct, action = self.get_action(self.env.rover_pos, self.env.target_pos)
                done = self.env.step(action)
                reward = self.env.reward()
                rov_new = self.env.get_rover_pos()
                tar_new = self.env.get_target_pos()
                Qxold = tar_old[0] * self.dim_x + rov_old[0]
                Qyold = tar_old[1] * self.dim_y + rov_old[1]
                Qxnew = tar_new[0] * self.dim_x + rov_new[0]
                Qynew = tar_new[1] * self.dim_y + rov_new[1]
                nextQmax = np.max(self.Qtable[Qxnew, Qynew, :])
                Qold = self.Qtable[Qxold, Qyold, direct]
                self.Qtable[Qxold, Qyold, direct] = (1 - self.alpha) * Qold + self.alpha \
                    * (reward + self.gamma * nextQmax)
                
            mem_buff[mem_iter] = self.env.timestep
            mem_iter += 1
            if i % 100 == 0:
                s = 'Iter: ' + str(i) + '\tMean: ' + str(np.mean(mem_buff)) + '\t Var: ' + str(np.var(mem_buff))
                print(s)
                mem_buff = np.zeros(100)
                mem_iter = 0
            if i % 1000 == 0:
                np.save(filename, self.Qtable)
                

if __name__ == '__main__':
    args = Parameters()
    env = Task_Rovers(args)
    learn = Qlearn(env)
    learn.train(50000)



    