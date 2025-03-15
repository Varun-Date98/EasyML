import pickle
import numpy as np
import pandas as pd
from typing import List

from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor,
                              GradientBoostingRegressor, AdaBoostRegressor)

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (OneHotEncoder, LabelEncoder, OrdinalEncoder, StandardScaler,
                                   MinMaxScaler, RobustScaler)
from sklearn.metrics import (classification_report, accuracy_score, precision_score, recall_score, r2_score,
                             root_mean_squared_error, mean_absolute_error, mean_squared_error)


# List of classification models
classification_models = ["Logistic Regression", "K-Neighbours Classifier", "Decision Tree",
                         "Gaussian NB", "SVC", "Random Forest", "Gradient Boosting Classifier"]

# List of regression models
regression_models = ["Linear Regression", "K-Neighbours Regressor", "Decision Tree Regressor",
                     "Ridge Regressor", "Lasso Regressor", "Random Forest Regressor",
                     "Gradient Boosting Regressor", "Ada Boost Regressor", "SVR"]

# List of metrics for classification tasks
classification_metrics = ["Classification Report", "Accuracy", "Precision", "Recall"]

# List of metrics for regression tasks
regression_metrics = ["RMSE", "MAE", "MSE", "R Squared"]

# List of supported scalers
scalers = ["Standard Scaler", "Min Max Scaler", "Robust Scaler"]

# List of supported encoders
encoders = ["One Hot Encoder", "Ordinal Encoder"]


class Engine:
    """Class to run ML pipeline on the data"""

    def __init__(self,
                 model: str,
                 data: pd.DataFrame,
                 features: List[str],
                 target: str,
                 task: str,
                 metric: str,
                 scaler: str,
                 encoder: str,
                 train_size: float = 0.8,
                 stratify: bool = False,
                 shuffle: bool = True):
        """
        Creates a new engine instance

        Args:
            model (str): Name of the model
            data (DataFrame): Input data datafame for the model
            features List[str]: List of feature columns
            target str: Column that needs to be predicted
            task str: One of 'Regression' or 'Classification'
            metric str: Evaluation metric for the model
            scaler str: Scaler to be used
            encoder str: Encoder to be used
            train_size float: Fraction of data to be used for training
            stratify bool: Whether to stratify target column while splitting. Default False.
            shuffle bool: Whether to shuffle data while splitting. Default True.
        """
        self.data = data
        self.target = target
        self.shuffle = shuffle
        self.model_name = model
        self.stratify = stratify
        self.train_size = train_size
        self.task = task
        self.metric = metric
        self.scaler = scaler
        self.encoder = encoder
        self.features = features

        self.model = None
        self.X_train = self.X_test = self.y_train = self.y_test = None

        self.__trained = False

    def _get_scaler(self):
        """
        Creates and returns the selected scaler object

        Returns:
            StandardScaler or MinMaxScaler or RobustScaler: An instance of the selected scikit learn scaler object
        """
        if self.scaler not in scalers:
            raise ValueError(f"Scaler should be one of {scalers}, found {self.scaler}")

        match self.scaler:
            case "Standard Scaler": scaler = StandardScaler()
            case "Min Max Scaler": scaler = MinMaxScaler()
            case "Robust Scaler": scaler = RobustScaler()
            case _: scaler = None

        return scaler

    def _get_encoder(self):
        """
        Creates and returns the selected encoder object

        Returns:
            OneHotEncoder or OrdinalEncoder: An instance of the selected scikit learn encoder
        """
        if self.encoder not in encoders:
            raise ValueError(f"Encoder should be one of {encoders}, found {self.encoder}")

        match self.encoder:
            case "One Hot Encoder": encoder = OneHotEncoder(handle_unknown="ignore")
            case "Ordinal Encoder": encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            case _: encoder = None

        return encoder

    def _get_y(self, target_col: pd.Series):
        """
        Method to get the Label encoded target column

        Args:
            target_col Series: Target column from the data

        Returns:
            Series: Label encoded target column
        """
        encoder = LabelEncoder()
        return encoder.fit_transform(target_col)

    def _get_model(self):
        """
        Creates and returns the selected scikit-learn model based on the task and model name

        Returns:
            sklearn.base.BaseEstimator: An instance of a scikit-learn model
        """
        if not (self.task in ["Regression", "Classification"]):
            raise ValueError(f"Task should be either Regression or Classification, got {self.task} instead")

        if not (self.model_name in classification_models or self.model_name in regression_models):
            raise ValueError("Not a valid model. Please select a valid model to train.")

        if self.task == "Classification":
            # Classification models
            match self.model_name:
                case "Logistic Regression": model = LogisticRegression()
                case "K-Neighbours Classifier": model = KNeighborsClassifier()
                case "Decision Tree": model = DecisionTreeClassifier()
                case "Gaussian NB": model = GaussianNB()
                case "SVC": model = SVC()
                case "Random Forest": model = RandomForestClassifier()
                case "Gradient Boosting Classifier": model = GradientBoostingClassifier()
                case _: model = None
        else:
            # Regression models
            match self.model_name:
                case "Linear Regression": model = LinearRegression()
                case "K-Neighbours Regressor": model = KNeighborsRegressor()
                case "Decision Tree Regressor": model = DecisionTreeRegressor()
                case "Ridge Regressor": model = Ridge()
                case "Lasso Regressor": model = Lasso()
                case "Random Forest Regressor": model = RandomForestRegressor()
                case "Gradient Boosting Regressor": model = GradientBoostingRegressor()
                case "Ada Boost Regressor": model = AdaBoostRegressor()
                case "SVR": model = SVR()
                case _: model = None

        return model

    def _preprocessing(self):
        """
        Performs preprocessing on the input data by splitting features and target, scaling numeric features, and encoding categorical features.

        Returns:
            tuple: A tuple containing:
                - X_train (array-like): Preprocessed training feature set.
                - X_test (array-like): Preprocessed test feature set.
                - y_train (array-like): Training target values.
                - y_test (array-like): Test target values.
        """
        X = self.data[self.features]
        y = self._get_y(self.data[self.target])

        stratify = y if self.stratify else None

        numeric_columns = []
        categorical_columns = []

        for col in X.columns:
            if np.issubdtype(X[col].dtype, np.number):
                numeric_columns.append(col)
            else:
                categorical_columns.append(col)

        scaler = self._get_scaler()
        encoder = self._get_encoder()

        num_pipe = Pipeline(steps=[
            ("scaler", scaler)
        ])

        cat_pipe = Pipeline(steps=[
            ("encoder", encoder)
        ])

        transformer = ColumnTransformer(transformers=[
            ("numeric", num_pipe, numeric_columns),
            ("categorical", cat_pipe, categorical_columns)
        ])

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=self.train_size,
                                                             shuffle=self.shuffle, stratify=stratify)

        X_train = transformer.fit_transform(X_train)
        X_test = transformer.transform(X_test)
        return X_train, X_test, y_train, y_test

    def train(self):
        """Trains the model after processing the training data"""
        try:
            self.model = self._get_model()

            if self.model is None:
                raise TypeError("Error occurred while getting model.")

            self.X_train, self.X_test, self.y_train, self.y_test = self._preprocessing()
        except ValueError:
            raise
        except TypeError:
            raise

        self.model.fit(self.X_train, self.y_train)
        self.__trained = True

    def get_metrics(self):
        """
        Computes and returns the evaluation metric for the trained model based on the task type.

        Returns:
            float or dict: The computed metric value. For example, a float value for Accuracy, Precision, etc., or a
            dictionary for the classification report when "Classification Report" is selected.
        """
        if not (self.metric in classification_metrics or self.metric in regression_metrics):
            raise ValueError("Not a valid metric. Please select a valid metric to train.")

        y_pred = self.model.predict(self.X_test)
        if self.task == "Classification":
            # Classification metrics
            match self.metric:
                case "Accuracy": return accuracy_score(y_true=self.y_test, y_pred=y_pred)
                case "Precision": return precision_score(y_true=self.y_test, y_pred=y_pred)
                case "Recall": return recall_score(y_true=self.y_test, y_pred=y_pred)
                case "Classification Report":
                    return classification_report(y_true=self.y_test, y_pred=y_pred, output_dict=True)
        else:
            # Regression metrics
            match self.metric:
                case "RMSE": return root_mean_squared_error(y_true=self.y_test, y_pred=y_pred)
                case "MAE": return mean_absolute_error(y_true=self.y_test, y_pred=y_pred)
                case "MSE": return mean_squared_error(y_true=self.y_test, y_pred=y_pred)
                case "R Squared": return r2_score(y_true=self.y_test, y_pred=y_pred)

    def save_model(self) -> str:
        """
        Serializes and saves the trained model to a file.

        This method pickles the current model and writes it to a file named "model.pkl".
        It then returns the file path where the model has been saved.

        Returns:
            str: The file path to the saved model.
        """
        model = self.model
        model_save_path = "model.pkl"

        if not self.__trained:
            raise AttributeError("The model has not been trained. Train the model before saving it.")

        with open(model_save_path, "wb") as f:
            pickle.dump(model, f)

        return model_save_path

    def is_trained(self):
        """
        Checks whether the model has been trained.

        Returns:
            bool: True if the model has been successfully trained, False otherwise.
        """
        return self.__trained
