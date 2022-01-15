import os.path

from stable_baselines3 import PPO

from TrackEnv2D import TrackEnv2D

env = TrackEnv2D(moving_target=True)


model = PPO.load(os.path.join("run_1","best_model.zip"),device='cpu')

obs = env.reset()
step = 0
rewards = []
while True:
    #action = env.action_space.sample()
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    step += 1
    env.render()
    #print(step)
    #print(reward)
    #print(done)
    rewards.append(reward)
    if done :
        print("##################")
        print(step)
        print(sum(rewards))
        rewards = []
        step = 0
        obs = env.reset()

        continue

