import gym
from baselines import deepq
import balance_bot
import numpy as np
import time
 
def callback(lcl, glb):
    # stop training if reward exceeds 199
    is_solved = False#lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved
 
def main():
    # create the environment
    env = gym.make("balancebot-v0", envStepCounterLimit = float("inf")) # <-- this we need to create
 
    # create the learning agent
    model = deepq.models.mlp([16, 16])
 
    # train the agent on the environment
    act = deepq.load("done.pkl")

    obs = env.reset()
    while True:
        action = act(np.array(obs)[None])[0]
        obs, rew, done, _ = env.step(action)
        if done:
            time.sleep(3)
            obs = env.reset()
            continue
 
if __name__ == '__main__':
    main()
