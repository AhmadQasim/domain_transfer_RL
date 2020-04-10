import yaml
import gym
import gym_baking.envs.utils as utils
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVR


class SupportVectorRegression:
    def __init__(self):
        self.config_path = "../reinforcemnet_learner/inventory.yaml"
        f = open(self.config_path, 'r')
        self.config = yaml.load(f, Loader=yaml.Loader)

        self.env = gym.make("gym_baking:Inventory-v0", config_path="../reinforcemnet_learner/inventory.yaml")

        self.items_to_id = utils.map_items_to_id(self.config)
        self.items_count = len(self.items_to_id.keys())
        self.steps = 1440
        self.days = 50
        self.bins_size = 60
        self.bins = int(self.steps / self.bins_size)
        self.windows_size = 3

        self.data = None
        self.bins_data = None

        self.items_svr = []
        self.C = 0.1
        self.epsilon = 1.0
        self.kernel = "rbf"
        self.degree = 2
        self.tol = 1e-9

    def sample_data(self):
        data = []

        for _ in range(self.days):
            orders = np.zeros(shape=(self.steps, self.items_count))
            prev_orders = np.zeros(shape=self.items_count)
            for j in range(self.steps):
                obs, reward, done, _ = self.env.step([0, 0])
                obs = utils.observation_state_vector(obs, return_count=True, items_to_id=self.items_to_id)
                curr_orders = obs[2]

                for key in curr_orders.keys():
                    order_val = curr_orders[key][0] - prev_orders[key]
                    prev_orders[key] = curr_orders[key][0]
                    orders[j, key] = order_val

            self.env.reset()
            data.append(orders)

        self.data = np.array(data)

        bins_data = np.zeros(shape=(self.days, self.bins, self.items_count))

        for i in range(self.days):
            for item in range(self.items_count):
                bins_sum = np.add.reduceat(self.data[i, :, item], range(0, self.steps, self.bins_size))
                bins_data[i, :, item] = bins_sum

        self.bins_data = bins_data

        return self.data, self.bins_data

    def prepare_data(self, Y, item):
        X = np.zeros(shape=(self.bins * self.days, self.windows_size, 2))
        for i in range(self.days):
            for j in range(self.bins):
                if j < self.windows_size:
                    continue
                X[(i * self.bins) + j, :, 0] = np.arange(j-self.windows_size, j)
                X[(i * self.bins) + j, :, 1] = Y[i, j - self.windows_size:j, item]

        return X.reshape((-1, self.windows_size * 2)), Y[:, :, item].reshape(-1, )

    def plot_data(self, data):
        fig = plt.figure()
        plt.plot(np.sum(data, axis=0) / self.days, alpha=0.5)
        plt.show()
        plt.close(fig)

    def train_svr(self):
        for item in range(self.items_count):
            _svr = SVR(C=self.C,
                       epsilon=self.epsilon,
                       gamma='auto',
                       kernel=self.kernel,
                       degree=self.degree,
                       tol=self.tol)

            X, Y = self.prepare_data(self.bins_data, item=item)
            _svr.fit(X, Y)
            self.items_svr.append(_svr)

    def test_svr(self):
        fig = plt.figure()

        for item in range(self.items_count):
            x_hats = []
            for i in range(self.windows_size, self.bins):
                x = self.bins_data[0, i-self.windows_size:i, item]
                x = np.vstack([np.arange(i-self.windows_size, i), x]).reshape(1, -1)
                x_hat = self.items_svr[item].predict(x)
                x_hats.append(x_hat)

            x_hats = np.array(x_hats)

            plt.plot(np.squeeze(x_hats), alpha=0.5)

        plt.show()
        plt.close(fig)


if __name__ == "__main__":
    svr = SupportVectorRegression()
    svr.sample_data()
    svr.plot_data(svr.bins_data)
    svr.train_svr()
    svr.test_svr()