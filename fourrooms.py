import numpy as np
from gym import core, spaces
from gym.envs.registration import register
import gym
import copy
import math

class Fourrooms(gym.Env):
    def __init__(self):
        layout ="""\
wwwwwwwwwwwww
w     w     w
w     w     w
w           w
w     w     w
w     w     w
ww wwww     w
w     www www
w     w     w
w     w     w
w           w
w     w     w
wwwwwwwwwwwww
"""
        
        self.occupancy = np.array([list(map(lambda c: 1 if c=='w' else 0, line)) for line in layout.splitlines()])
        # From any state the agent can perform one of four actions, up, down, left or right
        self.action_space = spaces.Discrete(100)
        self.observation_space = spaces.Discrete(np.sum(self.occupancy == 0))

        self.directions = [np.array((-1,0)), np.array((1,0)), np.array((0,-1)), np.array((0,1))]
        self.rng = np.random.RandomState(1234)

        self.n_actions = 100

        self.map_to_primitive = {k: self.rng.randint(4) for k in range(self.n_actions)}

        self.tostate = {}
        statenum = 0
        for i in range(self.occupancy.shape[0]):
            for j in range(self.occupancy.shape[1]):
                if self.occupancy[i, j] == 0:
                    self.tostate[(i,j)] = statenum
                    statenum += 1
        self.tocell = {v:k for k,v in self.tostate.items()}        
        self.goal = 91
        self.init_states = list(range(self.observation_space.n))
        print('-----------------------------------')
        print('GOAL IS AT:-',self.goal)
        print('------------------------------------')
        self.init_states.remove(self.goal)
        self.count = 0

    def empty_around(self, cell):
        avail = []
        for direction in self.directions:
            nextcell = tuple(cell + direction)
            if not self.occupancy[nextcell]:
                avail.append(nextcell)
        return avail

    def reset(self):
        state = self.rng.choice(self.init_states)
        self.currentcell = self.tocell[state]
        return state

    def step(self, action):
        """
        The agent can perform one of four actions,
        up, down, left or right, which have a stochastic effect. With probability 2/3, the actions
        cause the agent to move one cell in the corresponding direction, and with probability 1/3,
        the agent moves instead in one of the other three directions, each with 1/9 probability. In
        either case, if the movement would take the agent into a wall then the agent remains in the
        same cell.
        We consider a case in which rewards are zero on all state transitions.
        """
        action = self.map_to_primitive[action]
        nextcell = tuple(self.currentcell + self.directions[action])
        reward = 0
        if not self.occupancy[nextcell]:
            
            if self.rng.uniform() < (1/3.):
                empty_cells = self.empty_around(self.currentcell)
                self.currentcell = empty_cells[self.rng.randint(len(empty_cells))]
            else:
            	self.currentcell = nextcell
        else:
            reward = -1
        
        state = self.tostate[self.currentcell]
        if state ==self.goal:        
            done = True
            reward = 1
        else:
            done = False
        return state, reward, done

register(
    id='Fourrooms-v0',
    entry_point='fourrooms:Fourrooms',
    max_episode_steps=1000,
    reward_threshold=1,
)