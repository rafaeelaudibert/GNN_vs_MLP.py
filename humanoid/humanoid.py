import gym
import os
from baselines.ppo1 import mlp_policy, pposgd_simple
from humanoid_env import HumanoidEnv
import baselines.common.tf_util as U
import tensorflow as tf  # pylint: ignore-module

## Constants
NUM_TIMESTEPS = 300_000
MODEL_PATH = './saver/humanoid.tf'
 
# Copiado de https://github.com/openai/baselines/blob/master/baselines/ppo1/run_humanoid.py:40~45
class RewScale(gym.RewardWrapper):
    def __init__(self, env, scale):
        gym.RewardWrapper.__init__(self, env)
        self.scale = scale

    def reward(self, r):
        return r * self.scale

def callback(lcl, glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved

def save_state(fname):
    dirname = os.path.dirname(fname)
    if any(dirname):
        os.makedirs(dirname, exist_ok=True)
    saver = tf.train.Saver()
    saver.save(tf.get_default_session(), fname)

# Not used now
def load_state(fname):
    saver = tf.train.Saver()
    saver.restore(tf.get_default_session(), fname)

# Wrapper for mlp_policy.MlpPolicy
def policy_fn(name, ob_space, ac_space):
    return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
        hid_size=256, num_hid_layers=2)
 
def main():
    # Copiado de https://github.com/openai/baselines/blob/master/baselines/ppo1/run_humanoid.py
    U.make_session(num_cpu=1).__enter__()

    # create the environment
    env = HumanoidEnv()
    env = RewScale(env, 0.8)

    # train the agent on the environment
    pposgd_simple.learn(env, policy_fn,
        max_timesteps=NUM_TIMESTEPS,
        timesteps_per_actorbatch=2048,
        clip_param=0.1, entcoeff=0.0,
        optim_epochs=10,
        optim_stepsize=1e-4,
        optim_batchsize=64,
        gamma=0.99,
        lam=0.95,
        schedule='constant',
    )
    env.close()
    
    # Save trained model
    save_state(MODEL_PATH)
 
if __name__ == '__main__':
    main()

