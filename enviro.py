import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pyboy import PyBoy
from enum import Enum
import time
import sys
from PIL import Image
from skimage.transform import resize
from pyboy.utils import WindowEvent

actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
        ]
release_arrow = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP
        ]

release_button = [
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B
        ]
#actions = ['','a', 'b', 'left', 'right', 'up', 'down']

matrix_shape = (3,144,160)
game_area_observation_space = spaces.Box(low=0.0, high=255, shape=matrix_shape, dtype=np.uint8)
p_Message = ''


IMPlocations = {
            0: "Pallet Town",
            1: "Viridian City",
            2: "Pewter City",
            3: "Cerulean City",
            12: "Route 1",
            13: "Route 2",
            14: "Route 3",
            15: "Route 4",
            33: "Route 22",
            41: "Pokémon Center (Viridian City)",
            47: "Gate (Viridian City/Pewter City) (Route 2)",
            49: "Gate (Route 2)",
            50: "Gate (Route 2/Viridian Forest) (Route 2)",
            51: "viridian forest",
            58: "Pokémon Center (Pewter City)",
            59: "Mt. Moon (Route 3 entrance)",
            60: "Mt. Moon",
            61: "Mt. Moon",
            68: "Pokémon Center (Route 4)",
            193: "Badges check gate (Route 22)"
        }

class PkEnv(gym.Env):

    def __init__(self, maxlen):
        super().__init__()
        self.head = "null" if True else 'SDL2'
        self.pyboy = PyBoy("g1.gbc",window=self.head)
        with open("default.state", "rb") as f:
            self.pyboy.load_state(f)
        if self.head == "null":
            self.pyboy.set_emulation_speed(0)
        else:
            self.pyboy.set_emulation_speed(0)

        
        self._fitness=0
        self._previous_fitness=0
        self.step_count=0
        print("len:" ,maxlen)
        self.max_steps = maxlen
        self.locationsVisited = []
        self.impLocVisited = []

        self.action_space = spaces.Discrete(len(actions))
        self.observation_space = game_area_observation_space
        self.pyboy.game_wrapper.start_game()

        self.reset()
        
            

        
    


    def step(self, action):
        #if action != 0:
        #    self.pyboy.button(actions[action])
        self.pyboy.send_input(actions[action])
        #print(action)

        #self.pyboy.tick()
        self.step_count += 1
        self.updateLocVisit()

        done = self.step_count >= self.max_steps
        self._calculate_fitness()
        reward = self._fitness - self._previous_fitness

        obs = self.render()
        info = {}
        truncated = False
        for i in range(24):
            if i == 8:
                if action < 4:
                    # release arrow
                    self.pyboy.send_input(release_arrow[action])
                if action > 3 and action < 6:
                    # release button 
                    self.pyboy.send_input(release_button[action - 4])
                if actions[action] == WindowEvent.PRESS_BUTTON_START:
                    self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)
            if i % 8 and self.head !='null':
                self.pyboy.tick(render=True,sound=False)
            else:
                self.pyboy.tick(render=False,sound=False)
        
        return obs, reward*0.1, done, truncated, info


    def reset(self, **kwargs):
        print(self._fitness)
        
        with open("bestSave/maxfit.txt", "r") as f:
            TotalMax = float(f.readline())
        if self._fitness > TotalMax:
            TotalMax = self._fitness
            with open("bestSave/bestSave.state", "wb") as f:
                self.pyboy.save_state(f)
            with open("bestSave/maxfit.txt", "w") as f:
                f.write(str(TotalMax))
        
        
        with open("default.state", "rb") as f:
            self.pyboy.load_state(f)
        self.pyboy.game_wrapper.reset_game()
        self._fitness = 0
        self._previous_fitness = 0
        self.step_count = 0
        self.locationsVisited = []
        self.impLocVisited = []
        info = {}
        return self.render(), info


    def render(self):
        obs = self.pyboy.game_area()
        obs = (255*resize(obs, (3,144,160))).astype(np.uint8)
        return obs
    
    def close(self):
        self.pyboy.stop()

    def updateLocVisit(self):
        ypos = self.pyboy.memory[0xD360]
        xpos = self.pyboy.memory[0xD361]
        curLoc = [self.pyboy.memory[0xD35D],xpos,ypos]
        if curLoc not in self.locationsVisited:
            self.locationsVisited.append(curLoc)
        if self.pyboy.memory[0xD35D] not in self.impLocVisited and self.pyboy.memory[0xD35D] in IMPlocations:
            self.impLocVisited.append(self.pyboy.memory[0xD35D])

            print("\rMoved to " +IMPlocations[self.pyboy.memory[0xD35D]])

    def _calculate_fitness(self):
        self._previous_fitness=self._fitness
        #all_max_hp = sum(self.pyboy.memory[x] for x in [0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269])
        #all_cur_hp = sum(self.pyboy.memory[x] for x in [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248])
        all_lvls = sum(self.pyboy.memory[x] for x in [0xD18B, 0xD1B7, 0xD1E3, 0xD20F, 0xD23B, 0xD267])
        all_stats = sum(self.pyboy.memory[0xD18C:0xD196]) + sum(self.pyboy.memory[0xD1B8:0xD1C1]) + sum(self.pyboy.memory[0xD1E4:0xD1ED]) + sum(self.pyboy.memory[0xD210:0xD219]) + sum(self.pyboy.memory[0xD23C:0xD245]) + sum(self.pyboy.memory[0xD268:0xD271])
        all_exp = sum(self.pyboy.memory[0xD179:0xD17B]) + sum(self.pyboy.memory[0xD1A5:0xD1A7]) + sum(self.pyboy.memory[0xD1D1:0xD1D3]) + sum(self.pyboy.memory[0xD1FD:0xD1FF]) + sum(self.pyboy.memory[0xD229:0xD22B]) + sum(self.pyboy.memory[0xD255:0xD257])
        
        #all_stats = sum(self.pyboy.memory[x] for x in [0xD18C:0xD196, 0xD1B8:0xD1C1, 0xD1E4:0xD1ED, 0xD210:0xD219, 0xD23C:0xD245, 0xD268:D271])
#       battle_turn = pyboy.memory[0xD057]
#       p1_cur_hp = pyboy.memory[0xD16C]
        self._fitness=(all_stats*20) + (all_lvls * 100) + (len(self.locationsVisited) * 150) + (len(self.impLocVisited) * 500) + (all_exp*100)
        if self._fitness - self._previous_fitness < -1:
            print("\rfitness went down by " + str(self._previous_fitness - self._fitness))
        if self._fitness - self._previous_fitness > 200:
            print("\rfitness went up by " + str(self._fitness - self._previous_fitness))
        #print(self.step_count,self.max_steps)
        
        if self.step_count % 1000 == 0:
            percent = "{0:.2f}".format(100 * (self.step_count / float(self.max_steps)))
            filled_length = int(50 * self.step_count // self.max_steps)
            bar = '█' * filled_length + '-' * (50 - filled_length)
            print('\r%s |%s| %s%%' % ('Progress:', bar, percent))
        #sys.stdout.write('\r%s |%s| %s%%' % ('Progress:', bar, percent))

            #sys.stdout.flush()
            #print('|%s| %s%%' % (bar, percent), end="")