import yaml
import fire
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

        for i in range(50):
            obs, reward, done, _ = self.env.step(self.env.action_space.sample())
            obs = utils.observation_state_vector(obs, self.items_to_id, return_count=True)
            print(obs)


        exit(1)

        self.state_shape = self.env.observation_space.shape
        self.action_shape = self.env.action_space.shape
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.target_actor = Actor(self.state_shape, self.action_shape)
        self.target_critic = Critic(self.state_shape, self.action_shape)
        self.actor = Actor(self.state_shape, self.action_shape)
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
        self.noise = OrnsteinUhlenbeckProcess(size=self.action_shape)
        # range of the action possible for Pendulum-v0
        self.act_range = 2.0
        self.model_path = "models/DDPG-Pendulum.hdf5"

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

    def sample_from_memory(self):
        random_rows = np.random.randint(0, self.replay_buffer_states.shape[0], size=self.batch_size)
        return [self.replay_buffer_states[random_rows, :], self.replay_buffer_actions[random_rows, :],
                self.replay_buffer_rewards[random_rows, :], self.replay_buffer_done[random_rows, :],
                self.replay_buffer_next_states[random_rows, :]]

    def take_action(self, state):
        action = self.actor.forward(torch.tensor(state, dtype=torch.float))
        action = action.cpu().detach().numpy()
        new_observation, reward, done, info = self.env.step(action)
        return new_observation, action, reward, done

    def fill_empty_memory(self):
        observation = self.env.reset()
        for _ in range(100):
            new_observation, action, reward, done = self.take_action(observation)
            done = 1.0 if done else 0.0
            self.save_to_memory([torch.tensor(observation, dtype=torch.float).unsqueeze(0),
                                 torch.tensor(action, dtype=torch.float).unsqueeze(0),
                                 torch.tensor(reward, dtype=torch.float).unsqueeze(0).unsqueeze(0),
                                 torch.tensor(done, dtype=torch.float).unsqueeze(0).unsqueeze(0),
                                 torch.tensor(new_observation, dtype=torch.float).unsqueeze(0)
                                 ])
            if done:
                new_observation = self.env.reset()
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
        target_state_q_vals = self.target_critic.forward(next_states, target_actions)
        q_values = self.critic.forward(states, actions)
        q_targets = rewards + (self.discount_factor * target_state_q_vals)

        # update critic
        self.critic.zero_grad()
        critic_loss = self.critic_loss(q_values, q_targets)
        critic_loss.backward()
        self.critic_optim.step()

        # update actor
        self.actor.zero_grad()
        actor_loss = - self.critic.forward(states, self.actor.forward(states))
        actor_loss = actor_loss.mean()
        actor_loss.backward()
        self.actor_optim.step()

        # soft update weights
        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic, self.target_critic)

    def train(self):
        self.fill_empty_memory()
        total_reward = 0

        for ep in range(self.episodes):
            episode_rewards = []
            observation = self.env.reset()
            for step in range(self.max_steps):
                observation = np.squeeze(observation)
                new_observation, action, reward, done = self.take_action(observation)
                action = np.clip(action+self.noise.generate(step), -self.act_range, self.act_range)
                # action = action+self.noise.generate(step)

                self.save_to_memory([torch.tensor(observation, dtype=torch.float).unsqueeze(0),
                                     torch.tensor(action, dtype=torch.float).unsqueeze(0),
                                     torch.tensor(reward, dtype=torch.float).unsqueeze(0).unsqueeze(0),
                                     torch.tensor(done, dtype=torch.float).unsqueeze(0).unsqueeze(0),
                                     torch.tensor(new_observation, dtype=torch.float).unsqueeze(0)
                                     ])
                episode_rewards.append(reward)
                observation = new_observation
                self.optimize_model()

                self.epsilon = self.min_epsilon + (1 - self.min_epsilon) * np.exp(-self.eps_decay * ep)

                if done:
                    break

            # episode summary
            total_reward += np.sum(episode_rewards)
            print("Episode : ", ep)
            print("Episode Reward : ", np.sum(episode_rewards))
            print("Total Mean Reward: ", total_reward / (ep + 1))
            print("==========================================")

            torch.save(self.actor, self.model_path)

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
    def __init__(self, state_shape, action_shape):
        super(Actor, self).__init__()
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.fc1 = nn.Linear(self.state_shape[0], 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, self.action_shape[0])

        # initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))

        return x


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