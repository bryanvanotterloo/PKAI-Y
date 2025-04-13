import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pyboy import PyBoy
from enum import Enum
import time

actions = ['','a', 'b', 'left', 'right', 'up', 'down', 'start', 'select']

matrix_shape = (16, 20)
game_area_observation_space = spaces.Box(low=0, high=255, shape=matrix_shape, dtype=np.uint8)

class PkEnv(gym.Env):

    def __init__(self, pyboy, debug=False):
        super().__init__()
        self.pyboy = pyboy

        
        self._fitness=0
        self._previous_fitness=0
        self.step_count=0
        self.max_steps = 2000
        self.locationsVisited = []
        
        if not debug:
            self.pyboy.set_emulation_speed(0)

        self.action_space = spaces.Discrete(len(actions))
        self.observation_space = game_area_observation_space
        self.pyboy.game_wrapper.start_game()
        
            

        

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        if action == 0:
            pass
        else:
            self.pyboy.button(actions[action])
        
        self.pyboy.tick(sound=False)
        self.step_count += 1
        self.updateLocVisit()
        
        done = self.step_count >= self.max_steps
        #done = self.read_hp_fraction() == 0
        #done = self.pyboy.game_wrapper.game_over
        
        self._calculate_fitness()
        reward=self._fitness-self._previous_fitness

        observation=self.pyboy.game_area()
        info = {}
        truncated = False

        return observation, reward, done, truncated, info

    def reset(self, **kwargs):
        self.pyboy.game_wrapper.reset_game()
        self._fitness=0
        self._previous_fitness=0
        self.step_count=0
        self.max_steps = 2000
        self.locationsVisited = []

        observation=self.pyboy.game_area()
        info = {}
        return observation, info

    def render(self, mode='human'):
        pass

    def close(self):
        self.pyboy.stop()

    def updateLocVisit(self):
        ypos = self.pyboy.memory[0xD360]
        xpos = self.pyboy.memory[0xD361]
        curLoc = [self.pyboy.memory[0xD35D],xpos,ypos]
        if curLoc not in self.locationsVisited:
            self.locationsVisited.append(curLoc)

    def _calculate_fitness(self):
        self._previous_fitness=self._fitness
        all_max_hp = sum(self.pyboy.memory[x] for x in [0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269])
        all_cur_hp = sum(self.pyboy.memory[x] for x in [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248])
        all_lvls = sum(self.pyboy.memory[x] for x in [0xD18B, 0xD1B7, 0xD1E3, 0xD20F, 0xD23B, 0xD267])
        if all_max_hp == 0:
            all_max_hp = 1
#       battle_turn = pyboy.memory[0xD057]
#       p1_cur_hp = pyboy.memory[0xD16C]
        self._fitness=(all_cur_hp/all_max_hp*10) + (all_lvls * 5) + (len(self.locationsVisited) * 1)