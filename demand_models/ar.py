import gym_baking.envs.utils as utils
import matplotlib.pyplot as plt
import numpy as np
import gym
import yaml
from statsmodels.tsa.ar_model import AR
import pickle


class AutoRegression:
    def __init__(self,
                 model_path="../models/ar_models.pickle",
                 config_path="../reinforcement_learner/inventory.yaml",
                 steps=120,
                 days=10,
                 bins_size=10):

        self.config_path = config_path
        f = open(self.config_path, 'r')
        self.config = yaml.load(f, Loader=yaml.Loader)
        self.model_path = model_path

        self.env = gym.make("gym_baking:Inventory-v0", config_path="../reinforcement_learner/inventory.yaml")

        self.items_to_id = utils.map_items_to_id(self.config)
        self.items_count = len(self.items_to_id.keys())
        self.steps = steps
        self.days = days
        self.bins_size = bins_size
        self.bins = int(self.steps / self.bins_size)

        self.bins_smooth = self.steps - self.bins_size

        self.data = None
        self.prepared_data = []
        self.bins_data = None
        self.predictions = []

        self.prev = []

        self.model_fit = []

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
                    bins_sum = np.sum(self.data[i, j:j + self.bins_size, item])
                    bins_data[i, j, item] = bins_sum

        self.bins_data = bins_data

        return self.data, self.bins_data

    def prepare_data(self):
        for i in range(self.items_count):
            self.prepared_data.append(self.bins_data[:, :, i].reshape(self.days * self.bins_smooth))

        return self.prepared_data

    def train_ar(self):
        for i in range(self.items_count):
            model = AR(self.prepared_data[i])
            model_fit = model.fit(maxlag=self.bins_smooth,
                                  ic='t-stat',
                                  maxiter=35)

            self.model_fit.append(model_fit)
            self.prev.append(self.bins_data[-1, :self.model_fit[i].k_ar, i])

        return self.model_fit

    def test_ar(self):
        for i in range(self.items_count):
            predictions = self.model_fit[i].predict(start=self.prepared_data[i].shape[0],
                                                    end=self.prepared_data[i].shape[0] + self.bins_smooth - 1)
            self.predictions.append(predictions)

        return self.predictions

    def predict_next_n(self, curr_data, pred_steps, item):
        coeff = np.flip(self.model_fit[item].params)
        n_steps = curr_data.shape[0]

        curr_prev = self.prev[item].shape[0]
        self.prev[item] = np.append(self.prev[item], curr_data)

        curr_day_pred = self.prev[item].copy()
        for j in range(pred_steps):
            prediction = np.sum(coeff[:-1] * np.flip(curr_day_pred[n_steps + j:]))
            prediction += coeff[-1]
            curr_day_pred = np.append(curr_day_pred, prediction)

        average_diff = np.sum(curr_data - curr_day_pred[curr_prev + n_steps:curr_prev + pred_steps + n_steps]) / n_steps
        curr_day_pred += average_diff
        predictions = curr_day_pred[curr_prev + n_steps:curr_prev + pred_steps + n_steps]

        self.prev[item] = self.prev[item][n_steps:]

        return predictions

    def predict_day(self, n_steps, item):
        lag = self.model_fit[item].k_ar
        coeff = np.flip(self.model_fit[item].params)
        test = self.bins_data[-1, :n_steps, item + 1]

        for i in range(lag - n_steps):
            prediction = np.sum(coeff[:-n_steps + i] * test[i: lag + i])
            prediction += coeff[-1]

    def plot_real_data(self, data):
        fig = plt.figure()
        for i in range(self.items_count):
            plt.plot(np.sum(data[:, :, i], axis=0) / self.days, alpha=0.5,
                     label='Product ' + str(i))
        plt.title('Average Demand')
        plt.xlabel('Steps')
        plt.ylabel('Product Amount')
        plt.legend()
        plt.show()
        plt.close(fig)

    def plot_predicted_data(self, data):
        fig = plt.figure()
        for i in range(self.items_count):
            plt.plot(data[i], alpha=0.5, label="Prediction Product " + str(i))
        plt.title('Autoregressive Demand Prediction')
        plt.xlabel('Steps')
        plt.ylabel('Product Amount')
        plt.legend()
        plt.show()
        plt.close(fig)

    def save_models(self):
        model = {'ar_models': self.model_fit, 'prev': self.prev, 'item_count': self.items_count}

        with open(self.model_path, 'wb') as f:
            pickle.dump(model, f)

    def load_models(self):
        with open(self.model_path, 'rb') as f:
            model = pickle.load(f)
        self.items_count = model['item_count']

        for i in range(self.items_count):
            self.model_fit.append(model['ar_models'][i])
            self.prev.append(model['prev'][i])


if __name__ == "__main__":
    ar = AutoRegression("../models/ar_models.pickle")
    ar.sample_data()
    ar.prepare_data()
    ar.plot_real_data(ar.bins_data)
    ar.train_ar()
    training_pred = ar.test_ar()
    ar.plot_predicted_data(training_pred)
    ar.save_models()
    ar.load_models()

