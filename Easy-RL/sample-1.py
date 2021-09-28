import gym
import numpy as np


class SampleAgent():
    def __init__(self, env):
        pass

    def decide(self, observation):
        position, velocity = observation
        lb = min(-0.09 * (position+0.25) ** 2 + 0.03, 0.3 * (position + 0.9) ** 4 - 0.008)
        ub = -0.07 * (position + 0.38) ** 2 + 0.07
        if lb < velocity < ub:
            action = 2
        else:
            action = 0

        return action

    def learn(self, *args):
        pass


def play(env, agent, render=False, train=False):
    episode_reward = 0
    observation = env.reset()

    while True:
        if render:
            env.render()
        action = agent.decide(observation)
        next_observation, reward, done, _ = env.step(action)
        episode_reward += reward

        if train:
            agent.learn(observation, action, reward, done)
        if done:
            break
        observation = next_observation

    return episode_reward


env = gym.make('MountainCar-v0')
print(f"观测空间 = {env.observation_space}")
print(f"动作空间 = {env.action_space}")
print(f"观测范围 = {env.observation_space.low} - {env.observation_space.high}")
print(f"动作范围 = {env.action_space.n}")


agent = SampleAgent(env)
env.seed(0)

episode_reward = [play(env, agent, render=True) for _ in range(20)]
env.close()

print(episode_reward)
print(f"Mean Reward = {np.mean(episode_reward)}")