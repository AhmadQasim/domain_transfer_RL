import os
os.chdir("..")
from demand_models.ar import AutoRegression


class AutoRegressiveDemandPredictor:
    def __init__(self,
                 config_path,
                 steps,
                 days,
                 bins_size,
                 model_path=None,
                 load_model=False):

        """
        :param config_path: inventory.yaml file path
        :param steps: episode_max_steps in the inventory.yaml file
        :param days: number of days to sample for poisson for training
        :param bins_size: the bin size i.e. number of steps in each bin
        :param model_path: the path where the model is saved and loaded from
        :param load_model: load already saved model
        """

        self.model_path = model_path

        if load_model:
            self.ar = AutoRegression(model_path, config_path, steps, days, bins_size)
            self.ar.load_models()
        else:
            self.ar = AutoRegression(model_path, config_path, steps, days, bins_size)
            self.ar.train_ar()
            self.ar.save_models()

        self.lag = self.ar.model_fit.k_ar

    def predict(self, curr_data, pred_steps, item):
        """
        :param curr_data: the new data received
        :param pred_steps: number of predictions to be returned
        :param item: the product ID of curr_data and prediction data, according to inventory.yaml
        :return: predictions, numpy array
        """
        predictions = self.ar.predict_next_n(curr_data=curr_data, pred_steps=pred_steps, item=item)
        self.ar.save_models()

        return predictions
