import numpy as np
import pandas as pd
from typing import List

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, r2_score


classification_models = ["Logistic Regression", "K-Neighbours Classifier", "Decision Tree",
                         "Gaussian NB", "SVC", "Random Forest", "Gradient Boosting Classifier"]

regression_models = []


class Engine:
    def __init__(self,
                 model: str,
                 data: pd.DataFrame,
                 target: str,
                 train_size: float = 0.8,
                 stratify: bool = False,
                 shuffle: bool = True):
        self.data = data
        self.target = target
        self.shuffle = shuffle
        self.model_name = model
        self.stratify = stratify
        self.train_size = train_size

        self.model = None
        self.X_train = self.X_test = self.y_train = self.y_test = None

        self.__trained = False

    def _data_check(self):
        if not all(
            np.issubdtype(self.X_train[col].dtype, np.number) for col in self.X_train.columns
        ):
            raise ValueError("All columns must be of numeric type for model training. Found non-numeric columns.")

    def _train_test_split(self):
        X = self.data.drop(columns=self.target)
        y = self.data[self.target]

        X = self._prepare_features(X)
        stratify = y if self.stratify else None
        return train_test_split(X, y, train_size=self.train_size, stratify=stratify, shuffle=self.shuffle)

    def _get_model(self, model_name: str):
        if not (self.model_name in classification_models or self.model_name in regression_models):
            raise ValueError("Not a valid model. Please select a valid model to train.")

        model = LogisticRegression()

        match self.model_name:
            case "K-Neighbours Classifier": model = KNeighborsClassifier()
            case "Decision Tree": model = DecisionTreeClassifier()
            case "Gaussian NB": model = GaussianNB()
            case "SVC": model = SVC()
            case "Random Forest": model = RandomForestClassifier()
            case "Gradient Boosting Classifier": model = GradientBoostingClassifier()

        return model

    def _prepare_features(self, X: pd.DataFrame):
        numeric_cols = []
        categorical_cols = []

        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

        for col in X.columns:
            if np.issubdtype(X[col].dtype, np.number):
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)

        cat_features_out = encoder.fit_transform(X[categorical_cols])
        cat_columns_out = encoder.get_feature_names_out()
        cat_df = pd.DataFrame(cat_features_out, columns=cat_columns_out).reset_index(drop=True)
        num_df = X[numeric_cols].reset_index(drop=True)
        return pd.concat([num_df, cat_df], axis=1)

    def is_trained(self):
        return self.__trained

    def set_features(self, features: List[str]):
        columns_to_drop = []

        for col in self.data.columns:
            if col == self.target or col in features:
                continue

            columns_to_drop.append(col)

        self.data.drop(columns=columns_to_drop, inplace=True)
        return

    def train(self):
        self.model = self._get_model(self.model_name)
        self.X_train, self.X_test, self.y_train, self.y_test = self._train_test_split()

        try:
            self._data_check()
        except ValueError:
            raise

        self.model.fit(self.X_train, self.y_train)
        self.__trained = True

    def get_metrics(self):
        y_pred = self.model.predict(self.X_test)
        return classification_report(y_true=self.y_test, y_pred=y_pred, output_dict=True)
