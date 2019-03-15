import gym
from baselines import deepq
import balance_bot
 
def callback(lcl, glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved
 
def main():
    try:
        # create the environment
        env = gym.make("balancebot-v0", envStepCounterLimit = float("inf"))
 
        # create the learning agent
        model = deepq.models.mlp([32, 16])
 
        # train the agent on the environment
        act = deepq.learn(
            env, q_func=model, lr=1e-3,
            max_timesteps=200000, buffer_size=50000, exploration_fraction=0.1,
            exploration_final_eps=0.03, print_freq=5, checkpoint_freq=1000,callback=callback
        )
    except:
        pass
    finally:
        # Save trained model
        act.save("balance.pkl")
 
if __name__ == '__main__':
    main()
