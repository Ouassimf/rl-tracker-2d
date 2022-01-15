import os

import gym
import numpy as np
from matplotlib import pyplot as plt
from stable_baselines3.common.env_checker import check_env

from TrackEnv2D import TrackEnv2D
from stable_baselines3 import PPO,SAC,A2C,DQN

from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level.
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}")
                    self.model.save(self.save_path)

        return True

env = TrackEnv2D()
timesteps = 12250000

run_name = "run_mv_fast_target_w_dist_vector_no_abs"
log_dir = os.path.join(os.getcwd(),run_name)
os.makedirs(log_dir, exist_ok=True)
#check_env(env)
env = Monitor(env, log_dir)
#model = A2C.load("{}.gym".format(run_name))
model = PPO("MlpPolicy", env,verbose=1)
#model.load("{}.gym".format(run_name))
callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
model.learn(total_timesteps=timesteps,log_interval=1000,callback=callback)
model.save("{}.gym".format(run_name))

#plot_results([log_dir], timesteps, results_plotter.X_TIMESTEPS, "TD3 LunarLander")
#plt.show()

obs = env.reset()
step = 0
rewards = []
# while True:
#     action, _states = model.predict(obs)
#     obs, reward, done, info = env.step(action)
#     step += 1
#     #env.render()
#     #print(step)
#     #print(reward)
#     #print(done)
#     rewards.append(reward)
#     if done :
#         print("##################")
#         print(step)
#         print(sum(rewards))
#         rewards = []
#         step = 0
#         obs = env.reset()
#
#         continue


