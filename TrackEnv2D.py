import math

from gym import core, spaces
import matplotlib.pyplot as plt
import numpy as np


class TrackEnv2D(core.Env):
    def __init__(self, moving_target=False):

        # Define state space as x,y  -1 , +1
        self.moving_dir_sign = 1
        self.bonus = -1
        self.max_step = 250
        # (x,y) agent / (x,y) target / vect (x,y) agent <> target / vect target
        high = np.array([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0], dtype=np.float32)
        low = -high
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        self.step_distance = 0.03
        self.state = None
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.current_step = 0
        self.target_speed_factor = 1.5
        self.last_target_position = np.array([0.0,0.0])
        self.moving_target = moving_target
        # 1 => Horizontal / 2 => Vertical
        self.moving_dir = np.random.choice([1, 2], 1)
        # plt.show(block=False)
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        plt.ion()

    def step(self, action):
        done = False
        # Left

        self.move_agent(action)

        self.last_target_position = self.state[2:4]

        if self.moving_target:
            print("moving")
            self.move_target()

        self.state[4] = (self.state[2] - self.state[0])
        self.state[5] = (self.state[3] - self.state[1])

        done, reward = self.calculate_reward(done)

        if self.current_step > self.max_step:
            done = True
        info = {}
        self.current_step += 1
        # # move old obs to end of obs
        # self.state[6:] = self.state[:6]
        self.state[6] = self.state[2] - self.last_target_position[0]
        self.state[7] = self.state[3] - self.last_target_position[1]
        return self.state, reward, done, info

    def calculate_reward(self, done):
        bonus = self.bonus
        p1 = [self.state[0], self.state[1]]
        p2 = [self.state[2], self.state[3]]
        distance = math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))
        if distance < 0.05:
            bonus = 200
            done = True
        reward = -distance + bonus
        return done, reward

    def move_agent(self, action):
        if action == 0:
            if self.state[0] - self.step_distance < -1:
                self.state[0] = -1
            else:
                self.state[0] = self.state[0] - self.step_distance
        # Right
        elif action == 1:
            if self.state[0] + self.step_distance > 1:
                self.state[0] = 1
            else:
                self.state[0] = self.state[0] + self.step_distance
        elif action == 2:
            if self.state[1] - self.step_distance < -1:
                self.state[1] = -1
            else:
                self.state[1] = self.state[1] - self.step_distance
        # Up
        elif action == 3:
            if self.state[1] + self.step_distance > 1:
                self.state[1] = 1
            else:
                self.state[1] = self.state[1] + self.step_distance

    def render(self, **kwargs):

        x_coordinates = self.state[0]
        y_coordinates = self.state[1]
        target_x = self.state[2]
        target_y = self.state[3]
        plt.cla()
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        self.ax.scatter(target_x, target_y, color='red')
        self.ax.scatter(x_coordinates, y_coordinates, color='blue')
        self.fig.canvas.draw_idle()
        plt.pause(0.0005)

    def reset(self):
        self.current_step = 0
        self.state = self.observation_space.sample()
        self.state[4] = (self.state[2] - self.state[0])
        self.state[5] = (self.state[3] - self.state[1])
        #self.state[6:] = self.state[:6]
        self.state[6] = 0
        self.state[7] = 0
        self.moving_dir = np.random.choice([1, 2], 1)

        return self.state

    def move_target(self):
        # print(self.moving_dir)
        if self.moving_dir == 1:
            #   print("moving dir 1")
            next_pos = self.state[2] + self.moving_dir_sign * self.step_distance * self.target_speed_factor
            if next_pos > 1 or next_pos < -1:
                #      print("hit a wall")
                self.state[2] = 1 * self.moving_dir_sign
                self.moving_dir_sign = self.moving_dir_sign * (-1)
            else:
                # print(self.state[2])
                self.state[2] = next_pos
        elif self.moving_dir == 2:
            # print("moving dir 2")
            next_pos = self.state[3] + self.moving_dir_sign * self.step_distance * self.target_speed_factor
            if next_pos > 1 or next_pos < -1:
                #    print("hit a wall")
                self.state[3] = 1 * self.moving_dir_sign
                self.moving_dir_sign = self.moving_dir_sign * (-1)
            else:
                #   print(self.state[3])
                self.state[3] = next_pos
            #  print(self.state[3])
