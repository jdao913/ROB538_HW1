import random
from random import randint
from operator import add
import numpy as np
import math

class Task_Rovers:

    def __init__(self, parameters):
        self.params = parameters
        self.dim_x = parameters.dim_x
        self.dim_y = parameters.dim_y
        self.T = parameters.T
        self.nrollout = parameters.nrollout
        self.timestep = 0

        # Initialize food position container
        self.target_pos = [2, 10] # FORMAT: [item] = [x, y] coordinate

        # Initialize rover position container
        self.rover_pos = [1, 1]     # Track rover's position

        #Rover path trace (viz)
        self.rover_path = [(self.rover_pos[0], self.rover_pos[1])]
        self.action_seq = [0.0 for _ in range(self.params.action_dim)]

    def reset_rover(self, do_rand = True):
        if do_rand:
            self.rover_pos = [randint(1, self.dim_x), randint(1, self.dim_y)]
        else:
            self.rover_pos = [1, 1]

    def reset(self):
        self.reset_rover()
        self.rover_path = [(self.rover_pos[0], self.rover_pos[1])]
        # Reset Target
        self.target_pos = [2, 10]
        self.timestep = 0

    def step(self, action):
        self.timestep += 1

        self.rover_pos = list(map(add, self.rover_pos, action))
        # Check pos limits, make sure not out of bounds
        if self.rover_pos[0] < 1:
            self.rover_pos[0] = 1
        if self.rover_pos[0] > self.dim_x:
            self.rover_pos[0] = self.dim_x
        if self.rover_pos[1] < 1:
            self.rover_pos[1] = 1
        if self.rover_pos[1] > self.dim_y:
            self.rover_pos[1] = self.dim_y
        
        # Randomly move target
        # Moves [1,2,3,4] corresponds to up, right, down, left respectively
        moves = [1, 2, 3, 4]
        if self.target_pos[0] == 1:
            moves.remove(4)
        if self.target_pos[0] == self.dim_x:
            moves.remove(2)
        if self.target_pos[1] == 1:
            moves.remove(3)
        if self.target_pos[1] == self.dim_y:
            moves.remove(1)
        
        target_action = random.choice(moves)
        if target_action == 1:
            self.target_pos[1] += 1
        elif target_action == 2:
            self.target_pos[0] += 1
        elif target_action == 3:
            self.target_pos[1] -= 1
        else:
            self.target_pos[0] -= 1

    def reward(self):
        reward = -1*self.timestep
        if self.rover_pos == self.target_pos:
            reward += 20
        return reward

    def visualize(self):
        grid = [['-' for _ in range(5)] for _ in range(10)]

        # Draw in agengt
        x = int(rover_pos[0])
        y = int(rover_pos[1])
        #print x,y
        grid[x][y] = 'x'


        # Draw in target
        for loc, status in zip(self.poi_pos, self.poi_status):
            x = int(loc[0]); y = int(loc[1])
            marker = 'I' if status else 'A'
            grid[x][y] = marker

        for row in grid:
            print (row)
        print()

    def render(self):
        # Visualize
        grid = [['-' for _ in range(self.dim_x)] for _ in range(self.dim_y)]

        drone_symbol_bank = ["0", "1", '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
        # Draw in rover path
        for rover_id in range(self.params.num_rover):
            for time in range(self.params.num_timestep):
                x = int(self.rover_path[rover_id][time][0]);
                y = int(self.rover_path[rover_id][time][1])
                # print x,y
                grid[x][y] = drone_symbol_bank[rover_id]

        # Draw in food
        for loc, status in zip(self.poi_pos, self.poi_status):
            x = int(loc[0]);
            y = int(loc[1])
            marker = 'I' if status else 'A'
            grid[x][y] = marker

        for row in grid:
            print (row)
        print()

        print ('------------------------------------------------------------------------')