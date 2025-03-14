import os.path

import pandas as pd
from typing import List

import h2o
from h2o.automl import H2OAutoML


h2o.init()
exclude_list = ["DeepLearning", "XGBoost"]


class AutoML:
    def __init__(self,
                 data: pd.DataFrame,
                 feature_cols: List[str],
                 target: str,
                 train_size: float,
                 seed: int = 42):
        self.predictors = feature_cols
        self.target = target
        self.train_size = train_size
        self.seed = seed

        self.data = h2o.H2OFrame(data)

        self.aml = None
        self.train_set = self.test_set = None

        self.__trained = False

    def train(self):
        self.train_set, self.test_set = self.data.split_frame(ratios=[self.train_size], seed=self.seed)

        pipeline = H2OAutoML(max_models=5, seed=self.seed, max_runtime_secs=120, exclude_algos=exclude_list)
        pipeline.train(x=self.predictors, y=self.target, training_frame=self.train_set)
        self.aml = pipeline
        self.__trained = True

    def get_leaderboard(self):
        return self.aml.leaderboard.as_data_frame(use_multi_thread=True)

    def get_best_model(self):
        return self.aml.leader

    def get_metrics(self):
        return self.aml.leader.model_performance(self.test_set)

    def save_model(self, path: str) -> str:
        if not os.path.exists(path):
            raise ValueError(f"{path} does not exists. Provide a valid path to save the model.")

        model = self.aml.leader
        return h2o.save_model(model, path, force=True)

    def is_trained(self):
        return self.__trained