import os
from pathlib import Path
import datetime
import numpy as np
from pyboy import PyBoy
from stable_baselines3 import A2C, PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback
from gymnasium import Env, spaces
from enviro import PkEnv
from os.path import exists
import torch
import uuid
import multiprocessing
import math

from stable_baselines3.common.env_checker import check_env

# === Setup ===


def make_env(rank,maxlen):
    def _init():
        env = PkEnv(maxlen)
        return env
    return _init

#vec_env = DummyVecEnv([make_env])

#check_env(vec_env.envs[0])

if __name__ == '__main__':
    
    #episodes = 1000
    
    ep_length = 2048*3
    
    
    sess_path = Path(f'session_{str(uuid.uuid4())[:8]}')
    #for full power
    #cpu = multiprocessing.cpu_count() - 1
    
    #for low end
    cpu = math.floor(multiprocessing.cpu_count()/2) + 1
    
    

    env = SubprocVecEnv([make_env(i,ep_length) for i in range(cpu)])
    
    print(sess_path)
    checkpoint_callback = CheckpointCallback(save_freq=ep_length, save_path=sess_path, name_prefix='poke')

    learn_steps = 40
    #input this for min session load
    file_name = 'session_c2565952/poke_350208_steps' #'session_e41c9eff/poke_250871808_steps'
    
    if exists(file_name + '.zip'):
        print('\nloading checkpoint')
        model = PPO.load(file_name, env=env)
        model.n_steps = ep_length
        model.n_envs = cpu
        model.rollout_buffer.buffer_size = ep_length
        model.rollout_buffer.n_envs = cpu
        model.rollout_buffer.reset()
    else:
        model = PPO('CnnPolicy', env, verbose=1, n_steps=ep_length, batch_size=512, n_epochs=1, gamma=0.999)
    
    for i in range(learn_steps):
        print("step:",i)
        model.learn(total_timesteps=(ep_length)*cpu*1000, callback=checkpoint_callback)
    

"""
# Function to find the latest checkpoint in the checkpoints directory
def find_latest_checkpoint(save_dir):
    all_checkpoints = list(save_dir.glob('**/pk_net_*.pt'))
    if all_checkpoints:
        return max(all_checkpoints, key=os.path.getmtime)
    return None

base_save_dir = Path("checkpoints")
save_dir = base_save_dir / model_name #datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
head = "null" if True else 'SDL2'

latest_checkpoint = find_latest_checkpoint(save_dir)


#print(TotalMax)

def runAI(key):

    pyboy = PyBoy("g1.gbc",window=head)
    pyboy.set_emulation_speed(2.0)
    with open("default.state", "rb") as f:
        pyboy.load_state(f)
    env=PkEnv(pyboy)
    
    check_env(env)
    obs, info = env.reset()
    done = False
    matrix_shape = (4, 16, 20)
    pk_agent = PkAgent(state_dim=matrix_shape, action_dim=env.action_space.n, save_dir=save_dir,env=env,max_x=16,max_y=20)
    latest_checkpoint = find_latest_checkpoint(save_dir)
    if latest_checkpoint:
        print("{:<80}".format(f"Found latest checkpoint at {latest_checkpoint}. Resuming from this checkpoint."))
        pk_agent.load(latest_checkpoint)
    else:
        save_dir.mkdir(parents=True, exist_ok=True)
        print("{:<80}".format("No existing checkpoints found. Created a new directory for this training session."))

    
    while True:
        
        action = pk_agent.get_action(obs)
        
        
        # Agent performs action
        next_obs, reward, terminated, truncated, info = env.step(action)

        # Remember
        #pokemon_pinball_agent.cache(state, next_state, action, reward, done)

        # Learn
        pk_agent.update(obs,action,reward,done,next_obs)

        # Logging
        #logger.log_step(reward, loss, q)

        # Update state
        obs = next_obs
        # Check if end of game
        if terminated:
            print("{:<80}".format("\r" + str(env._fitness)))
            with open("bestSave/maxfit.txt", "r") as f:
                TotalMax = float(f.readline())
            if env._fitness > TotalMax:
                TotalMax = env._fitness
                with open("bestSave/bestSave.state", "wb") as f:
                    pyboy.save_state(f)
                with open("bestSave/maxfit.txt", "w") as f:
                    f.write(str(TotalMax))
            break
    return [env._fitness,pk_agent.q_values]
    


#pyboy = PyBoy("pinball.gbc",game_wrapper=False)



# wrappers
#env=SkipFrame(env, skip=4)
#env=FrameStack(env, num_stack=4)
if __name__ == '__main__':
    

    current_episode = 0



    episodes = 1000
    print("{:<80}".format("Starting from episode " + str(current_episode)))
    while current_episode < episodes:
        print("{:<80}".format(str(current_episode)))
        
        results = []
        maxFit = 0
        #TopFit = 0
        maxLoc = 0
        counter = 0
        # Play the game!    
        with multiprocessing.Pool(processes=(multiprocessing.cpu_count() - 1)) as pool:
            #pool.map(runAI, args=(current_episode))
            results = pool.map(runAI, range(7))
            print("{:<80}".format(str(results)))
        for x in results:
            if x[0] > maxFit:
                maxFit = x[0]
                maxLoc = counter
            print("{:<80}".format(str(x[0])))
            counter += 1
        print("{:<80}".format(str(results[maxLoc])))
        torch.save(results[maxLoc][1], f"{save_dir}/pk_net_{current_episode}.pt")


        #logger.log_episode()
        #pk_agent.save(save_dir)
        #if (current_episode % 20 == 0) or (current_episode == episodes - 1):
        #    logger.record(episode=current_episode, epsilon=pokemon_pinball_agent.exploration_rate, step=pokemon_pinball_agent.curr_step)
        current_episode+=1
        #pk_agent.curr_episode = current_episode"""