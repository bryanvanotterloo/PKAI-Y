model_name= "Resume_working"




from Agent import PkAgent
from enviro import PkEnv
from pathlib import Path
from collections import deque
import random, datetime, os
from gymnasium import spaces
import numpy as np
from pyboy import PyBoy

# Function to find the latest checkpoint in the checkpoints directory
def find_latest_checkpoint(save_dir):
    all_checkpoints = list(save_dir.glob('**/pk_net_*.chkpt'))
    if all_checkpoints:
        return max(all_checkpoints, key=os.path.getmtime)
    return None

pyboy = PyBoy("g1.gbc")
with open("default.state", "rb") as f:
    pyboy.load_state(f)
#pyboy = PyBoy("pinball.gbc",game_wrapper=False)

env=PkEnv(pyboy)

# wrappers
#env=SkipFrame(env, skip=4)
#env=FrameStack(env, num_stack=4)


#use_cuda = torch.cuda.is_available()
#print(f"Using CUDA: {use_cuda}")


base_save_dir = Path("checkpoints")
save_dir = base_save_dir / model_name #datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

matrix_shape = (4, 16, 20)
pk_agent = PkAgent(state_dim=matrix_shape, action_dim=env.action_space.n, save_dir=save_dir,env=env)

latest_checkpoint = find_latest_checkpoint(save_dir)

current_episode = 0

if latest_checkpoint:
    print(f"Found latest checkpoint at {latest_checkpoint}. Resuming from this checkpoint.")
    pk_agent.load(latest_checkpoint)
    #logger = MetricLogger(save_dir,resume=True)
    current_episode = pk_agent.curr_episode
else:
    save_dir.mkdir(parents=True, exist_ok=True)
    #logger = MetricLogger(save_dir)
    print("No existing checkpoints found. Created a new directory for this training session.")



episodes = 100
print("Starting from episode",current_episode)
while current_episode < episodes:
    print(current_episode)
    obs, info = env.reset()
    done = False

    # Play the game!
    while True:


        
        # Run agent on the state
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
            print(env._fitness)
            break

    #logger.log_episode()
    pk_agent.save(save_dir)
    #if (current_episode % 20 == 0) or (current_episode == episodes - 1):
    #    logger.record(episode=current_episode, epsilon=pokemon_pinball_agent.exploration_rate, step=pokemon_pinball_agent.curr_step)
    current_episode+=1
    pk_agent.curr_episode = current_episode