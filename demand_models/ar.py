import gym_baking.envs.utils as utils
import matplotlib.pyplot as plt
import numpy as np
import gym
import yaml
from statsmodels.tsa.ar_model import AR


class AutoRegression:
    def __init__(self):
        self.config_path = "../reinforcemnet_learner/inventory.yaml"
        f = open(self.config_path, 'r')
        self.config = yaml.load(f, Loader=yaml.Loader)

        self.env = gym.make("gym_baking:Inventory-v0", config_path="../reinforcemnet_learner/inventory.yaml")

        self.items_to_id = utils.map_items_to_id(self.config)
        self.items_count = len(self.items_to_id.keys())
        self.steps = 120
        self.days = 10
        self.bins_size = 10
        self.bins = int(self.steps / self.bins_size)

        self.bins_smooth = self.steps - self.bins_size

        self.data = None
        self.prepared_data = None
        self.bins_data = None
        self.predictions = None

        self.model_fit = None

        self.train_prod = 0
        self.test_prod = 1

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

        bins_data = np.zeros(shape=(self.days, self.bins_smooth, self.items_count))

        for i in range(self.days):
            for j in range(self.bins_smooth):
                for item in range(self.items_count):
                    bins_sum = np.sum(self.data[i, j:j+self.bins_size, item])
                    bins_data[i, j, item] = bins_sum

        self.bins_data = bins_data

        return self.data, self.bins_data

    def prepare_data(self):
        self.prepared_data = self.bins_data[:, :, self.train_prod].reshape(self.days * self.bins_smooth)

        return self.prepared_data

    def train_ar(self):
        model = AR(self.prepared_data)
        model_fit = model.fit(maxlag=self.bins_smooth,
                              ic='t-stat',
                              maxiter=35)

        self.model_fit = model_fit

        return self.model_fit

    def test_ar(self):
        predictions = self.model_fit.predict(start=self.prepared_data.shape[0],
                                             end=self.prepared_data.shape[0] + self.bins_smooth - 1)
        self.predictions = predictions

        fig = plt.figure()
        plt.plot(predictions, alpha=0.5)
        plt.show()
        plt.close(fig)

        return self.predictions

    def sample_ar(self):
        lag = self.model_fit.k_ar
        coeff = np.flip(self.model_fit.params)

        test = self.bins_data[-1, :lag, self.test_prod]
        prev = self.bins_data[-2, :lag, self.train_prod]

        n_steps = 70
        prev = np.append(prev, test[0:n_steps])
        predictions = test[0:n_steps]

        for i in range(lag-n_steps):
            prediction = np.sum(coeff[:-1] * prev[i+n_steps: lag+i+n_steps])
            prediction += coeff[-1]

            predictions = np.append(predictions, prediction)
            prev = np.append(prev, prediction)

        predictions = np.array(predictions)
        predictions += -np.amin(predictions)

        fig = plt.figure()
        plt.plot(test, alpha=0.5)
        plt.plot(predictions, alpha=0.5)
        plt.show()
        plt.close(fig)

    def predict_day(self, n_steps):
        lag = self.model_fit.k_ar
        coeff = np.flip(self.model_fit.params)
        test = self.bins_data[-1, :n_steps, self.test_prod]

        for i in range(lag-n_steps):
            prediction = np.sum(coeff[:-n_steps+i] * test[i: lag + i])
            prediction += coeff[-1]

    def plot_data(self, data):
        fig = plt.figure()
        plt.plot(np.sum(data, axis=0) / self.days, alpha=0.5)
        plt.show()
        plt.close(fig)


if __name__ == "__main__":
    ar = AutoRegression()
    ar.sample_data()
    ar.prepare_data()
    ar.plot_data(ar.bins_data)
    ar.train_ar()
    ar.test_ar()
    ar.sample_ar()
