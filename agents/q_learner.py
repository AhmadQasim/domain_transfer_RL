import fire
import gym
import numpy as np
import gym_baking
import gym_baking.envs.utils as utils

from agents.base_agent import BaseAgent


class QLearningDiscreteEnv(BaseAgent):
    def __init__(self, env_name='FrozenLake-v0'):
        super().__init__()
        self.env = gym.make("gym_baking:Inventory-v0", config_path="../reinforcemnet_learner/inventory.yaml")
        self.observation_space = {'producer_state': {'production_queue': [], 'is_busy': False},
                                  'inventory_state': {'products': []},
                                  'consumer_state': {'order_queue': []}}
        obs, reward, done, _ = self.env.step(self.env.action_space.sample())
        print()
        exit(1)
        self.q_table = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        self.episodes = 15000
        self.max_actions_per_episode = 100
        self.epsilon = 1
        self.min_epsilon = 0.01
        self.eps_decay = 0.005
        self.learning_rate = 0.8
        self.discount_factor = 0.95
        self.rewards = []
        self.test_eps = 3000
        self.test_rewards = []

        print("Running Environment: ", env_name)

    def train(self):
        for i in range(self.episodes):
            print("Episode: ", i)
            observation = self.env.reset()
            total_reward_per_episode = 0
            for a in range(self.max_actions_per_episode):
                # take random action
                if np.random.rand() < self.epsilon:
                    action = self.env.action_space.sample()
                # take best action
                else:
                    action = np.argmax(self.q_table[observation, :])
                q_value = self.q_table[observation, action]

                # take action
                new_observation, reward, done, info = self.env.step(action)
                new_state_q_value = np.max(self.q_table[new_observation, :])

                # update q-table
                self.q_table[observation, action] = q_value + self.learning_rate * (reward + (self.discount_factor *
                                                                                              new_state_q_value -
                                                                                              q_value))

                # track reward per episode
                total_reward_per_episode += reward

                # update state
                observation = new_observation
                if done:
                    break
            self.epsilon = self.min_epsilon + (1 - self.min_epsilon) * np.exp(-self.eps_decay * i)
            self.rewards.append(total_reward_per_episode)
        self.env.close()

        print("Average reward: ", sum(self.rewards) / self.episodes)

    def test(self):
        # test agent
        for i in range(self.test_eps):
            observation = self.env.reset()
            total_reward_per_episode = 0
            for _ in range(self.max_actions_per_episode):
                action = np.argmax(self.q_table[observation, :])
                new_observation, reward, done, info = self.env.step(action)
                total_reward_per_episode += reward
                observation = new_observation
                if done:
                    break
            self.test_rewards.append(total_reward_per_episode)

        print("Average reward for test agent: ", sum(self.test_rewards) / self.test_eps)


if __name__ == "__main__":
    fire.Fire(QLearningDiscreteEnv)
