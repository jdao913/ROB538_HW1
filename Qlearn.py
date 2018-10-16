import random
from random import randint
import numpy as np
import math
from grid import * 
import time
import matplotlib.pyplot as plt

class Qlearn:

    def __init__(self, env):
        self.env = env
        self.dim_x = self.env.dim_x
        self.dim_y = self.env.dim_y
        self.eps = .1       #Epsilon to act epsilon greedy by
        self.alpha = .1     #Learning rate
        self.gamma = .8     #Discount factor
        # Intialize Q table with zeros; Qtable[action][x][y]
        # Q value for state s where agent at (xa, ya) and target at (xt, yt) and action a:
        # Qtable[xt*dim_x + xa, yt*dim_y + ya, a]
        # Action 0, 1, 2, 3, correspond to up, right, down, left respectively
        self.Qtable = [np.zeros((self.dim_x**2, self.dim_y**2, self.env.params.action_dim)) 
                        for i in range(env.nrover)]
        # Make impossible actions -infty value
        # for i in range(env.nrover):
        #     for j in range(self.dim_x**2):
        #         for k in range(self.dim_y):
        #             self.Qtable[i][j, k*self.dim_y, 0] = -np.inf
        #             self.Qtable[i][j, (k + 1) * self.dim_y - 1, 2] = -np.inf
        #     for j in range(self.dim_y**2):
        #         for k in range(self.dim_x):
        #             self.Qtable[i][k*self.dim_x, j, 3] = -np.inf
        #             self.Qtable[i][(k + 1) * self.dim_x - 1, j, 1] = -np.inf

    #Compute epsilon greedy action according to current Q values
    def get_action(self, rover_pos, target_pos):
        direct = [0 for i in range(self.env.nrover)]
        action = [[0, 0] for i in range(self.env.nrover)]
        for i in range(self.env.nrover):
            if random.random() > self.eps:        # Act greedily
                actionVs = self.Qtable[i][target_pos[0]*self.dim_x + rover_pos[i][0],
                                    target_pos[1]*self.dim_y + rover_pos[i][1], :]
                direct[i] = np.argmax(actionVs)
            else:       # Random action
                moves = [0, 1, 2, 3]
                if rover_pos[i][0] == 0:
                    moves.remove(3)
                if rover_pos[i][0] == self.dim_x - 1:
                    moves.remove(1)
                if rover_pos[i][1] == 0:
                    moves.remove(2)
                if rover_pos[i][1] == self.dim_y - 1:
                    moves.remove(0)
                direct[i] = random.choice(moves)
            action[i] = [-1, 0]
            if direct[i] == 0:
                action[i] = [0, 1]
            elif direct[i] == 1:
                action[i] = [1, 0]
            elif direct[i] == 2:
                action[i] = [0, -1]
        return direct, action
    
    def eval(self, niter):
        mem_buff = np.zeros(niter)
        for i in range(niter):
            done = False
            self.env.reset()
            while not done:
                direct, action = self.get_action(self.env.rover_pos, self.env.target_pos)
                done = self.env.step(action)
            mem_buff[i] = self.env.timestep
        return mem_buff

    def train(self, niter, filename = "Qtest.npy"):
        eval_score = np.zeros(int(niter / 100))
        eval_iter = 0
        # Initialize data structs
        rov_old = [[0, 0] for i in range(self.env.nrover)]
        rov_new = [[0, 0] for i in range(self.env.nrover)]
        Qxold = [0 for i in range(self.env.nrover)]
        Qyold = [0 for i in range(self.env.nrover)]
        Qxnew = [0 for i in range(self.env.nrover)]
        Qynew = [0 for i in range(self.env.nrover)]
        nextQmax = [0 for i in range(self.env.nrover)]
        Qold = [0 for i in range(self.env.nrover)]
        for n in range(niter):
            self.env.reset()
            done = False
            while not done:
                # Save old positions
                for i in range(self.env.nrover):
                    rov_old[i] = self.env.rover_pos[i]
                tar_old = self.env.target_pos
                # Get action according to current Q's and take step
                direct, action = self.get_action(self.env.rover_pos, self.env.target_pos)
                done = self.env.step(action)
                # reward = self.env.reward()
                reward = self.env.multireward()
                # Save new positions
                tar_new = self.env.get_target_pos()
                for i in range(self.env.nrover):
                    rov_new[i] = self.env.rover_pos[i]
                    # Find indices for Q values
                    Qxold[i] = tar_old[0] * self.dim_x + rov_old[i][0]
                    Qyold[i] = tar_old[1] * self.dim_y + rov_old[i][1]
                    Qxnew[i] = tar_new[0] * self.dim_x + rov_new[i][0]
                    Qynew[i] = tar_new[1] * self.dim_y + rov_new[i][1]
                    # Update Q table
                    nextQmax[i] = np.max(self.Qtable[i][Qxnew[i], Qynew[i], :])
                    Qold[i] = self.Qtable[i][Qxold[i], Qyold[i], direct[i]]
                    self.Qtable[i][Qxold[i], Qyold[i], direct[i]] = (1 - self.alpha) * Qold[i] + self.alpha \
                            * (reward[i] + self.gamma * nextQmax[i])
            if n % 100 == 0:
                evals = self.eval(1000)
                eval_score[eval_iter] = np.mean(self.eval(100))
                s = 'Iter: ' + str(n) + '\tMean: ' + str(eval_score[eval_iter]) + '\t Var: ' + str(np.var(evals))
                print(s)
                eval_iter += 1
            if n % 1000 == 0:
                np.save(filename, self.Qtable)
        return eval_score
                

if __name__ == '__main__':
    t1 = time.time()
    args = Parameters()
    args.nrover = 1
    print("Training enviroment with " + str(args.nrover) + " agents")
    env = Task_Rovers(args)
    learn = Qlearn(env)
    eval_scores = learn.train(100000, "SingleSum.npy")
    np.save("multi_eval.npy", eval_scores)
    plt.plot(eval_scores)
    # plt.show()
    # plt.ylim(top=50)
    plt.savefig("SingleAgent.png")
    




    