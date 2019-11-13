import yaml
import fire
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from utils import OrnsteinUhlenbeckProcess
import gym_baking.envs.utils as utils

from agents.base_agent import BaseAgent

"""
Implementation of Deep Deterministic Policy Gradients on A2C with TD-0 value returns
"""

torch.set_default_tensor_type('torch.cuda.FloatTensor')


class DDPG(BaseAgent):
    def __init__(self):
        super().__init__()
        self.config_path = "../reinforcemnet_learner/inventory.yaml"
        f = open(self.config_path, 'r')
        self.config = yaml.load(f, Loader=yaml.Loader)

        self.env = gym.make("gym_baking:Inventory-v0", config_path="../reinforcemnet_learner/inventory.yaml")
        self.observation_space = {'producer_state': {'production_queue': [], 'is_busy': False},
                                  'inventory_state': {'products': []},
                                  'consumer_state': {'order_queue': []}}

        self.items_to_id = utils.map_items_to_id(self.config)
        self.items_count = len(self.items_to_id.keys())
        self.feature_count = 5

        self.state_shape = (self.items_count * self.feature_count, )
        self.action_shape = self.env.action_space.shape

        '''
        for i in range(5):
            obs, reward, done, _ = self.env.step(self.env.action_space.sample())
            print(self.env.action_space.sample())
            obs = utils.observation_state_vector(obs, return_count=True, items_to_id=self.items_to_id)
            print(obs)

        exit(1)
        '''

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.target_actor = Actor(self.state_shape, self.action_shape, self.items_count)
        self.target_critic = Critic(self.state_shape, self.action_shape)
        self.actor = Actor(self.state_shape, self.action_shape, self.items_count)
        self.critic = Critic(self.state_shape, self.action_shape)
        self.replay_buffer_states = torch.zeros(size=(1, self.state_shape[0]))
        self.replay_buffer_actions = torch.zeros(size=(1, self.action_shape[0]))
        self.replay_buffer_rewards = torch.zeros(size=(1, 1))
        self.replay_buffer_done = torch.zeros(size=(1, 1))
        self.replay_buffer_next_states = torch.zeros(size=(1, self.state_shape[0]))
        self.replay_buffer_size_thresh = 100000
        self.batch_size = 64
        self.episodes = 5000
        self.max_steps = 500
        self.test_episodes = 1000
        self.discount_factor = 0.99
        self.test_rewards = []
        self.epochs = 10
        self.tau = 1e-3
        self.epsilon = 1
        self.min_epsilon = 0.01
        self.eps_decay = 0.005
        self.default_q_value_actor = -1
        self.noise = OrnsteinUhlenbeckProcess(size=1)
        # range of the action possible for Pendulum-v0
        self.act_range = 2.0
        self.model_path = "../models/inventory_agent_ddpg.hdf5"

        # models
        self.actor_optim = torch.optim.Adam(self.actor.parameters())
        self.critic_optim = torch.optim.Adam(self.critic.parameters())
        self.critic_loss = nn.MSELoss()
        self.hard_update(self.actor, self.target_actor)
        self.hard_update(self.critic, self.target_critic)

    def save_to_memory(self, experience):
        if self.replay_buffer_states.shape[0] > self.replay_buffer_size_thresh:
            self.replay_buffer_states = self.replay_buffer_states[1:, :]
            self.replay_buffer_actions = self.replay_buffer_actions[1:, :]
            self.replay_buffer_rewards = self.replay_buffer_rewards[1:, :]
            self.replay_buffer_done = self.replay_buffer_done[1:, :]
            self.replay_buffer_next_states = self.replay_buffer_next_states[1:, :]
        self.replay_buffer_states = torch.cat([self.replay_buffer_states, experience[0]])
        self.replay_buffer_actions = torch.cat([self.replay_buffer_actions, experience[1]])
        self.replay_buffer_rewards = torch.cat([self.replay_buffer_rewards, experience[2]])
        self.replay_buffer_done = torch.cat([self.replay_buffer_done, experience[3]])
        self.replay_buffer_next_states = torch.cat([self.replay_buffer_next_states, experience[4]])

    def prepare_obs_for_model_in(self, observation):
        # [in_production_count, inventory_count, age_mean, consumer_count, waiting_times_mean]
        result = np.zeros(shape=(self.items_count, self.feature_count))

        production_queue = observation[0]
        inventory = observation[1]
        consumer_queue = observation[2]

        for key, val in production_queue.items():
            result[key, 0] = val

        for key, val in inventory.items():
            result[key, 1] = val[0]
            result[key, 2] = val[1]

        for key, val in consumer_queue.items():
            result[key, 3] = val[0]
            result[key, 4] = val[1]

        return result

    def preprocess_observation(self, observation):
        observation = utils.observation_state_vector(observation,
                                                     return_count=True,
                                                     items_to_id=self.items_to_id)
        observation = self.prepare_obs_for_model_in(observation)

        return observation

    def sample_from_memory(self):
        random_rows = np.random.randint(0, self.replay_buffer_states.shape[0], size=self.batch_size)
        return [self.replay_buffer_states[random_rows, :], self.replay_buffer_actions[random_rows, :],
                self.replay_buffer_rewards[random_rows, :], self.replay_buffer_done[random_rows, :],
                self.replay_buffer_next_states[random_rows, :]]

    @staticmethod
    def preprocess_action(prob, count):
        items_type = torch.argmax(prob, dim=1)
        items_type = items_type.cpu().detach().numpy()
        count = count.cpu().detach().numpy()
        action = np.column_stack(tup=(items_type, count))

        return action

    def take_action(self, state):
        prob, count = self.actor.forward(torch.tensor(state, dtype=torch.float))
        action = self.preprocess_action(prob, count).flatten().astype(np.int16)
        new_observation, reward, done, info = self.env.step(action)
        new_observation = self.preprocess_observation(new_observation)

        # for all items, reward is inversely related to mean_age and mean_waiting_times
        for i in range(self.items_count):
            reward -= new_observation[i, 3] # + (new_observation[i, 2]

        items = torch.distributions.Categorical(prob)
        items = torch.tensor(items.sample(), dtype=torch.float).unsqueeze(1)
        target_actions = torch.cat([items, count], dim=1)

        # print(new_observation, target_actions, reward)
        # print("-------")
        return new_observation.flatten(), target_actions, reward, done

    def fill_empty_memory(self):
        observation = self.env.reset()
        observation = self.preprocess_observation(observation)
        observation = np.expand_dims(observation.flatten(), axis=0)
        for _ in range(100):
            new_observation, action, reward, done = self.take_action(observation)
            new_observation = np.expand_dims(new_observation, axis=0)
            done = 1.0 if done else 0.0
            self.save_to_memory([torch.tensor(observation, dtype=torch.float),
                                 torch.tensor(action, dtype=torch.float),
                                 torch.tensor(reward, dtype=torch.float).unsqueeze(0).unsqueeze(0),
                                 torch.tensor(done, dtype=torch.float).unsqueeze(0).unsqueeze(0),
                                 torch.tensor(new_observation, dtype=torch.float)
                                 ])
            if done:
                new_observation = self.env.reset()
                new_observation = self.preprocess_observation(new_observation)
                new_observation = np.expand_dims(new_observation.flatten(), axis=0)

            observation = new_observation

    def soft_update(self, source, target):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    @staticmethod
    def hard_update(source, target):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def optimize_model(self):
        states, actions, rewards, done, next_states = self.sample_from_memory()

        target_actions = self.target_actor.forward(next_states)
        items = torch.distributions.Categorical(target_actions[0])
        items = torch.tensor(items.sample(), dtype=torch.float).unsqueeze(1)
        target_actions = torch.cat([items, target_actions[1]], dim=1)
        target_state_q_vals = self.target_critic.forward(next_states,
                                                         target_actions)
        q_values = self.critic.forward(states, actions)
        q_targets = rewards + (self.discount_factor * target_state_q_vals)

        # update critic
        self.critic.zero_grad()
        critic_loss = self.critic_loss(q_values, q_targets)
        critic_loss.backward()
        self.critic_optim.step()

        # update actor
        target_actions = self.actor.forward(states)
        self.actor.zero_grad()
        items = torch.distributions.Categorical(target_actions[0])
        items = torch.tensor(items.sample(), dtype=torch.float).unsqueeze(1)
        target_actions = torch.cat([items, target_actions[1]], dim=1)
        actor_loss = - self.critic.forward(states, target_actions)
        actor_loss = actor_loss.mean()
        actor_loss.backward()
        self.actor_optim.step()

        # soft update weights
        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic, self.target_critic)

    def train(self):
        self.fill_empty_memory()
        total_reward = 0

        total_mean_reward = []

        for ep in range(self.episodes):
            episode_rewards = []
            observation = self.env.reset()
            observation = self.preprocess_observation(observation)
            observation = np.expand_dims(observation.flatten(), axis=0)
            for step in range(self.max_steps):
                new_observation, action, reward, done = self.take_action(observation)
                action = action.cpu().detach().numpy()
                new_observation = np.expand_dims(new_observation, axis=0)
                action[0, 1] = np.clip(action[0, 1]+self.noise.generate(step), -self.act_range, self.act_range)
                # action = action+self.noise.generate(step)

                self.save_to_memory([torch.tensor(observation, dtype=torch.float),
                                     torch.tensor(action, dtype=torch.float),
                                     torch.tensor(reward, dtype=torch.float).unsqueeze(0).unsqueeze(0),
                                     torch.tensor(done, dtype=torch.float).unsqueeze(0).unsqueeze(0),
                                     torch.tensor(new_observation, dtype=torch.float)
                                     ])
                episode_rewards.append(reward)
                observation = new_observation
                self.optimize_model()

                self.epsilon = self.min_epsilon + (1 - self.min_epsilon) * np.exp(-self.eps_decay * ep)

                if done:
                    break

            # episode summary
            total_reward += np.sum(episode_rewards)
            total_mean_reward.append(total_reward / (ep + 1))
            print("Episode : ", ep)
            print("Episode Reward : ", np.sum(episode_rewards))
            print("Total Mean Reward: ", total_reward / (ep + 1))
            print("==========================================")

            torch.save(self.actor, self.model_path)

        plt.plot(list(range(self.episodes)), total_mean_reward)
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        plt.legend()
        plt.show()

    def test(self):
        # test agent
        actor = torch.load(self.model_path)
        for i in range(self.test_episodes):
            observation = np.asarray(list(self.env.reset()))
            total_reward_per_episode = 0
            while True:
                self.env.render()
                action = actor.forward(torch.tensor(observation, dtype=torch.float))
                new_observation, reward, done, info = self.env.step(action.cpu().detach().numpy())
                total_reward_per_episode += reward
                observation = new_observation
                if done:
                    break
            self.test_rewards.append(total_reward_per_episode)

        print("Average reward for test agent: ", sum(self.test_rewards) / self.test_episodes)


class Actor(nn.Module):
    def __init__(self, state_shape, action_shape, items_count):
        super(Actor, self).__init__()
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.fc1 = nn.Linear(self.state_shape[0], 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc_count = nn.Linear(128, 1)
        self.fc_type = nn.Linear(128, items_count)

        # initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc_count.weight)
        nn.init.xavier_uniform_(self.fc_type.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        item_type = torch.softmax(self.fc_type(x), dim=1)
        count = torch.sigmoid(self.fc_count(x)) * 30

        return item_type, count


class Critic(nn.Module):
    def __init__(self, state_shape, action_shape):
        super(Critic, self).__init__()
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.fc1_state = nn.Linear(self.state_shape[0], 256)
        self.fc1_action = nn.Linear(self.action_shape[0], 256)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

        # initialize weights
        nn.init.xavier_uniform_(self.fc1_state.weight)
        nn.init.xavier_uniform_(self.fc1_action.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, state, action):
        x1 = state
        x2 = action

        x1 = F.relu(self.fc1_state(x1))
        x2 = F.relu(self.fc1_action(x2))

        x = torch.cat([x1, x2], dim=1)

        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


if __name__ == '__main__':
    fire.Fire(DDPG)