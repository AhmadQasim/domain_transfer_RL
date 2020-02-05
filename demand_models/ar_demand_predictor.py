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
        predictions = self.ar.predict_next_n(curr_data=curr_data, pred_steps=pred_steps, item=item)
        self.ar.save_models()

        return predictions
