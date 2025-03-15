import pandas as pd
from typing import List

import h2o
from h2o.automl import H2OAutoML


# Initialize H2O
h2o.init()
exclude_list = ["DeepLearning", "XGBoost"]


class AutoML:
    def __init__(self,
                 data: pd.DataFrame,
                 feature_cols: List[str],
                 target: str,
                 train_size: float,
                 seed: int = 42):
        """
        Creates an AutoML object
        Args:
            data (DataFrame): Input data for modeling
            feature_cols (List[str]): List of feature columns
            target (str): Target column to be predicted
            train_size (float): Fraction of data to be used for training
            seed (int): Random seed for reproducibility
        """
        self.predictors = feature_cols
        self.target = target
        self.train_size = train_size
        self.seed = seed

        self.data = h2o.H2OFrame(data)

        self.aml = None
        self.train_set = self.test_set = None

        self.__trained = False

    def train(self):
        """Trains the H2O AutoML pipeline on the provided dataset"""
        self.train_set, self.test_set = self.data.split_frame(ratios=[self.train_size], seed=self.seed)

        pipeline = H2OAutoML(max_models=5, seed=self.seed, max_runtime_secs=120, exclude_algos=exclude_list)
        pipeline.train(x=self.predictors, y=self.target, training_frame=self.train_set)
        self.aml = pipeline
        self.__trained = True

    def get_leaderboard(self):
        """Returns H2O Auto ML leader board as a pandas DataFrame"""
        return self.aml.leaderboard.as_data_frame(use_multi_thread=True)

    def get_best_model(self):
        """Returns the best model from the H2O Auto ML leader board"""
        return self.aml.leader

    def get_metrics(self):
        """Returns metrics for the H2O Auto ML pipeline"""
        return self.aml.leader.model_performance(self.test_set)

    def save_model(self) -> str:
        """
        Saves the model in the current directory and returns its path

        Returns:
            str: path where the model is saved
        """
        model = self.aml.leader
        return h2o.save_model(model, path="./", force=True)

    def is_trained(self):
        """
        Checks whether the model has been trained.

        Returns:
            bool: True if the model has been successfully trained, False otherwise.
        """
        return self.__trained