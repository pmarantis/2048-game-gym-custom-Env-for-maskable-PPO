import gym 
from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete 
import numpy as np
import random
import os
import logic
from typing import Callable
import math

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

##LEFT = 0
##UP = 1
##RIGHT = 2
##DOWN = 3

class Env2048(Env):
    def __init__(self):
        self.action_space = Discrete(4)
        self.observation_space = Box(0, 18, (4 * 4,), dtype=np.uint8)
        self.state = np.array(logic.start_game())
        self.mat = self.state.copy()
        
        #apply log2 to state
        for i in range(4):
            for j in range(4):
                if self.state[i][j] != 0:
                    self.state[i][j] = int(math.log2(self.state[i][j]))
                
        self.state = self.state.flatten()
        
    def step(self,action):

        #reward = 0
        info = {}
                
        #change to state
        if action == 0:
            self.mat, flag, reward = logic.move_left(self.mat)
            status = logic.get_current_state(self.mat)

            if(status == 'GAME NOT OVER'):
                done = False
                logic.add_new_2_or_4(self.mat)
            else:
                done = True

        elif action == 1:
            self.mat, flag, reward = logic.move_up(self.mat)
            status = logic.get_current_state(self.mat)

            if(status == 'GAME NOT OVER'):
                done = False
                logic.add_new_2_or_4(self.mat)
            else:
                done = True

        elif action == 2:
            self.mat, flag, reward = logic.move_right(self.mat)
            status = logic.get_current_state(self.mat)

            if(status == 'GAME NOT OVER'):
                done = False
                logic.add_new_2_or_4(self.mat)
            else:
                done = True
      
        elif action == 3:
            self.mat, flag, reward = logic.move_down(self.mat)
            status = logic.get_current_state(self.mat)

            if(status == 'GAME NOT OVER'):
                done = False
                logic.add_new_2_or_4(self.mat)
            else:
                done = True

        self.state = np.array(self.mat)
        for i in range(4):
            for j in range(4):
                if self.state[i][j] != 0:
                    self.state[i][j] = int(math.log2(self.state[i][j]))
        self.state = self.state.flatten()
        
        return self.state,reward, done, info

    def render(self,mode = "human"):
        logic.game_print(self.mat)
        print('\n')
    
    def reset(self):
        # Reset game mat
        self.state = np.array(logic.start_game())
        self.mat = self.state.copy()
        self.state = self.state.flatten()
        return self.state

    def valid_action_mask(self):
        action_masks = np.zeros((4,), dtype=int)
        
        flag_left = logic.check_left(self.mat)
        if flag_left == True:
            action_masks[0] = 1
        
        flag_right = logic.check_right(self.mat)
        if flag_right == True:
            action_masks[2] = 1

        flag_up = logic.check_up(self.mat)
        if flag_up == True:
            action_masks[1] = 1

        flag_down = logic.check_down(self.mat)
        if flag_down == True:
            action_masks[3] = 1

        return action_masks
    
    def num_envs(self):
        return 1


def mask_fn(env: gym.Env) -> np.ndarray:
    # Do whatever you'd like in this function to return the action mask
    # for the current env. In this example, we assume the env has a
    # helpful method we can rely on.
    return env.valid_action_mask()

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

env = Env2048()
env = ActionMasker(env, mask_fn)

log_path = os.path.join('Training', 'Logs')
model_save_name = 'PPO_2048'
PPO_Path = os.path.join('Training', 'Saved Models', model_save_name)

def train_agent(log_path, total_timesteps, PPO_Path):
    log_path = log_path
    model = MaskablePPO(MaskableActorCriticPolicy, env,
                        learning_rate=linear_schedule(0.001),
                        verbose=1, tensorboard_log=log_path)
    model.learn(total_timesteps=total_timesteps)
    model.save(PPO_Path)

def test_trained_agent(PPO_Path, n_episodes):
    model = MaskablePPO.load(PPO_Path, env=env) #reload model
    episodes = n_episodes
    for episode in range(1, episodes+1):
        state = env.reset()
        done = False
        score = 0 
        
        while not done:
            #env.render()
            action_masks = get_action_masks(env)
            action, _state = model.predict(state, action_masks=action_masks)
            state, reward, done, info = env.step(action)
            score+=reward
        print(f'max number is {pow(2,max(state))}')
        #print('Episode:{} Score:{}'.format(episode, score))
    env.close()

def retrain_saved_model(log_path, total_timesteps, PPO_Path):
    learning_rate=linear_schedule(0.001)
    custom_objects = { 'learning_rate': learning_rate }
    model = MaskablePPO.load(PPO_Path, env=env, custom_objects=custom_objects)
    model.learn(total_timesteps=total_timesteps)
    model.save(PPO_Path)

#train_agent(log_path, 5_000_000, PPO_Path)
test_trained_agent(os.path.join('Training', 'Saved Models', 'PPO_2048_1'), 100)
#retrain_saved_model(log_path, 5_000_000, PPO_Path)


###CHANGE NET ARCH
##
##net_arch=dict(pi=[1024,1024,512,128], vf=[1024,1024,512,256])
##model = MaskablePPO(MaskableActorCriticPolicy, env, learning_rate=linear_schedule(0.01),
##                    gamma=0.95, verbose=1, tensorboard_log=log_path,
##                    policy_kwargs={'net_arch':net_arch})
##model.learn(total_timesteps=5_000_000)
##model.save('PPO_2048')






