import random
from random import randint
from operator import add
import numpy as np
import math
import time

class Parameters:
    def __init__(self):

        #Rover domain
        self.dim_x = 10
        self.dim_y = 5 #HOW BIG IS THE ROVER WORLD
        self.action_dim = 4
        self.T = 100
        self.nrollout = 10
        self.obs_radius = 15 #OBSERVABILITY FOR EACH ROVER
        self.act_dist = 1.5 #DISTANCE AT WHICH A POI IS CONSIDERED ACTIVATED (OBSERVED) BY A ROVER
        self.angle_res = 30 #ANGLE RESOLUTION OF THE QUASI-SENSOR THAT FEEDS THE OBSERVATION VECTOR
        self.num_poi = 10 #NUM OF POIS
        self.num_rover = 4 #NUM OF ROVERS
        self.num_timestep = 25 #TIMESTEP PER EPISODE

class Task_Rovers:

    def __init__(self, parameters):
        self.params = parameters
        self.dim_x = parameters.dim_x
        self.dim_y = parameters.dim_y
        self.T = parameters.T
        self.nrollout = parameters.nrollout
        self.timestep = 0

        # Initialize food position container
        self.target_pos = [9, 3] # FORMAT: [item] = [x, y] coordinate

        # Initialize rover position container
        self.rover_pos = [0, 0]     # Track rover's position


    def reset(self):
        # self.rover_pos = [0, 0]
        new_pos = [9, 3]
        while new_pos == [9, 3]:
            new_pos[0] = randint(0, self.dim_x - 1)
            new_pos[1] = randint(0, self.dim_y - 1)
        self.rover_pos = new_pos
        # Reset Target
        self.target_pos = [9, 3]
        self.timestep = 0

    def get_target_pos(self):
        return self.target_pos
    
    def get_rover_pos(self):
        return self.rover_pos

    def rand_action(self, pos):
        # Moves [0, 1, 2, 3] corresponds to up, right, down, left respectively
        moves = [0, 1, 2, 3]
        if pos[0] == 0:
            moves.remove(3)
        if pos[0] == self.dim_x - 1:
            moves.remove(1)
        if pos[1] == 0:
            moves.remove(2)
        if pos[1] == self.dim_y - 1:
            moves.remove(0)
        direct = random.choice(moves)
        move = [-1, 0]
        if direct == 0:
            move = [0, 1]
        elif direct == 1:
            move = [1, 0]
        elif direct == 2:
            move = [0, -1]
        return move

    def check_goal(self):
        if self.rover_pos == self.target_pos:
            return True
        else:
            return False

    def step(self, action):
        self.timestep += 1

        self.rover_pos = list(map(add, self.rover_pos, action))
        # Check pos limits, make sure not out of bounds
        if self.rover_pos[0] < 0:
            self.rover_pos[0] = 0
        if self.rover_pos[0] >= self.dim_x:
            self.rover_pos[0] = self.dim_x - 1
        if self.rover_pos[1] < 0:
            self.rover_pos[1] = 0
        if self.rover_pos[1] >= self.dim_y:
            self.rover_pos[1] = self.dim_y - 1
        
        if self.check_goal():
            return True

        # Randomly move target
        targ_act = self.rand_action(self.target_pos)
        self.target_pos[0] += targ_act[0]
        self.target_pos[1] += targ_act[1]

        return self.check_goal()

    def reward(self):
        # reward = -1*self.timestep
        reward = -1
        if self.rover_pos == self.target_pos:
            reward = 20
        return reward

    def visualize(self):
        grid = [['-' for _ in range(self.dim_x)] for _ in range(self.dim_y)]

        # Draw in agengt
        x = int(self.rover_pos[0])
        y = int(self.rover_pos[1])
        #print x,y
        grid[y][x] = 'x'


        # Draw in target
        # for loc, status in zip(self.poi_pos, self.poi_status):
        #     x = int(loc[0]); y = int(loc[1])
        #     marker = 'I' if status else 'A'
        #     grid[x][y] = marker
        x = int(self.target_pos[0])
        y = int(self.target_pos[1])
        grid[y][x] = "T"

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
                x = int(self.rover_path[rover_id][time][0])
                y = int(self.rover_path[rover_id][time][1])
                # print x,y
                grid[x][y] = drone_symbol_bank[rover_id]

        # Draw in food
        for loc, status in zip(self.poi_pos, self.poi_status):
            x = int(loc[0])
            y = int(loc[1])
            marker = 'I' if status else 'A'
            grid[x][y] = marker

        for row in grid:
            print (row)
        print()

        print ('------------------------------------------------------------------------')

if __name__ == '__main__':
    args = Parameters()
    env = Task_Rovers(args)
    for i in range(20):
        env.visualize()
        updown = random.choice([0, 1])
        leftright = random.choice([-1, 1])
        act = [updown, leftright]
        env.step(act)



