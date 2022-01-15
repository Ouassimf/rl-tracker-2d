import os.path

from stable_baselines3 import A2C,PPO

from TrackEnv2D import TrackEnv2D

env = TrackEnv2D(moving_target=True)

#model = A2C("MlpPolicy", env,verbose=1)
model = PPO.load(os.path.join("run_mv_fast_target_w_dist_vector_no_abs","best_model.zip"))

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

