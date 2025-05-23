import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pyboy import PyBoy
from pyboy.utils import WindowEvent
from skimage.transform import resize

pyboy = PyBoy('g1.gbc')
#with open("default.state", "rb") as f:
#    pyboy.load_state(f)
    
with open("bestSave/bestSave.state", "rb") as f:
    pyboy.load_state(f)
    
maxfit = 0

with open("bestSave/maxfit.txt", "r") as f:
    maxfit = float(f.readline())
print(maxfit)
locations = {
            0: "Pallet Town",
            1: "Viridian City",
            2: "Pewter City",
            3: "Cerulean City",
            12: "Route 1",
            13: "Route 2",
            14: "Route 3",
            15: "Route 4",
            33: "Route 22",
            37: "Red house first",
            38: "Red house second",
            39: "Blues house",
            40: "oaks lab",
            41: "Pokémon Center (Viridian City)",
            42: "Poké Mart (Viridian City)",
            43: "School (Viridian City)",
            44: "House 1 (Viridian City)",
            47: "Gate (Viridian City/Pewter City) (Route 2)",
            49: "Gate (Route 2)",
            50: "Gate (Route 2/Viridian Forest) (Route 2)",
            51: "viridian forest",
            52: "Pewter Museum (floor 1)",
            53: "Pewter Museum (floor 2)",
            54: "Pokémon Gym (Pewter City)",
            55: "House with disobedient Nidoran♂ (Pewter City)",
            56: "Poké Mart (Pewter City)",
            57: "House with two Trainers (Pewter City)",
            58: "Pokémon Center (Pewter City)",
            59: "Mt. Moon (Route 3 entrance)",
            60: "Mt. Moon",
            61: "Mt. Moon",
            68: "Pokémon Center (Route 4)",
            193: "Badges check gate (Route 22)"
        }

        


t = 0
locationsVisited = []
pyboy.set_emulation_speed(0)

while t < 10000:
    pyboy.tick()
    t+=1
    if t == 100:
        t = 0
        mapLoc = locations[pyboy.memory[0xD35D]]
        p1_max_hp = pyboy.memory[0xD023:0xD024]
        battle_turn = pyboy.memory[0xD057]
        p1_cur_hp = pyboy.memory[0xD16C]
        p1_id = pyboy.memory[0xD16A]
        all_lvls = sum(pyboy.memory[x] for x in [0xD18B, 0xD1B7, 0xD1E3, 0xD20F, 0xD23B, 0xD267])
        ypos = pyboy.memory[0xD360]
        xpos = pyboy.memory[0xD361]
        curLoc = [pyboy.memory[0xD35D],xpos,ypos]
        if curLoc not in locationsVisited:
            locationsVisited.append(curLoc)
        print(locationsVisited)
        obs = pyboy.game_area()
        obs = (255*resize(obs, (3,144,160))).astype(np.uint8)
        #battle_name = pyboy.memory[0xCFD9:0xCFE3]
        #battle_name = "".join([chr(x-63) for x in battle_name if x != 0x50])
        #enemy_name = pyboy.memory[0xD008:0xD012]
        #print(enemy_name)
        #enemy_name = "".join([chr(x-63) for x in enemy_name if x > 0])
        #all_cur_hp = sum(pyboy.memory[x] for x in [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248])
        all_stats = sum(pyboy.memory[0xD18C:0xD196])
        all_exp = sum(pyboy.memory[0xD179:0xD17B]) + sum(pyboy.memory[0xD1A5:0xD1A7]) + sum(pyboy.memory[0xD1D1:0xD1D3]) + sum(pyboy.memory[0xD1FD:0xD1FF]) + sum(pyboy.memory[0xD229:0xD22B]) + sum(pyboy.memory[0xD255:0xD257])
        print(mapLoc,all_lvls,all_stats, "EXP:",all_exp)
    pass

#with open("bestSave/maxfit.txt", "w") as f:
#    f.write(str(maxfit + 1))

#with open("default.state", "wb") as f:
#    pyboy.save_state(f)
pyboy.stop()