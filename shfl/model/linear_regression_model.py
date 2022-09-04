from shfl.model.model import TrainableModel
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import metrics


class LinearRegressionModel(TrainableModel):
    """
    This class offers support for scikit-learn linear regression model. It implements [TrainableModel](../model/#trainablemodel-class)

    # Arguments:
        n_features: number of features (independent variables)
        n_targets: number of targets to predict (default is 1)
    """
    def __init__(self, n_features, n_targets=1):
        self._check_initialization(n_features)
        self._check_initialization(n_targets)
        self._model = LinearRegression()
        self._n_features = n_features
        self._n_targets = n_targets
        self.set_model_params([np.zeros(n_targets), np.zeros((n_targets, n_features))])
        
    def train(self, data, labels):
        """
        Implementation of abstract method of class [TrainableModel](../model/#trainablemodel-class)

        # Arguments
            data: Data, array-like of shape (n_samples, n_features)
            labels: Target, array-like of shape (n_samples,) or (n_samples, n_targets)
        """
        self._check_data(data)
        self._check_labels(labels)
        
        self._model.fit(data, labels)

    def predict(self, data):
        """
        Implementation of abstract method of class [TrainableModel](../model/#trainablemodel-class)

        # Arguments:
            data: Data, array-like of shape (n_samples, n_features)

        # Returns:
            prediction: array with predictions for data argument
        """
        self._check_data(data)
        
        prediction = self._model.predict(data)
        
        return prediction
    
    def evaluate(self, data, labels):
        """
        Implementation of abstract method of class [TrainableModel](../model/#trainablemodel-class)
        Metrics for evaluating model's performance.
        
        # Arguments:
            data: Data, array-like of shape (n_samples, n_features)
            labels: Target, array-like of shape (n_samples,) or (n_samples, n_targets)

        # Returns:
            rmse: RMSE value for the prediction. [MSE](
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html)
            r2: R2 value for the prediction. [R2](
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html)
        """
        self._check_data(data)
        self._check_labels(labels)
        
        prediction = self.predict(data)
        rmse = np.sqrt(metrics.mean_squared_error(labels, prediction))
        r2 = metrics.r2_score(labels, prediction)
        
        return rmse, r2

    def performance(self, data, labels):
        """
        Implementation of abstract method of class [TrainableModel](../model/#trainablemodel-class)

        # Arguments:
            data: Data, array-like of shape (n_samples, n_features)
            labels: Target, array-like of shape (n_samples,) or (n_samples, n_targets)

        # Returns:
            negative_rmse: Negative RMSE value for the prediction
        """
        self._check_data(data)
        self._check_labels(labels)

        prediction = self.predict(data)
        rmse = np.sqrt(metrics.mean_squared_error(labels, prediction))

        return -rmse

    def get_model_params(self):
        """
        Implementation of abstract method of class [TrainableModel](../Model/#trainablemodel-class)
        """
        
        return [self._model.intercept_, self._model.coef_]

    def set_model_params(self, params):
        """
        Implementation of abstract method of class [TrainableModel](../Model/#trainablemodel-class)
        """
        
        self._model.intercept_ = params[0]
        self._model.coef_ = params[1]
            
    def _check_data(self, data):
        """
        Method that checks if the data dimension if correct.

        # Arguments:
            data: array with data
        """
        if data.ndim == 1:
            if self._n_features != 1:
                raise AssertionError("Data need to have the same number of features described by the model "
                                     + str(self._n_features) + ". Current data have only 1 feature.")
        elif data.shape[1] != self._n_features:
            raise AssertionError("Data need to have the same number of features described by the model "
                                 + str(self._n_features) + ". Current data has " + str(data.shape[1]) + " features.")

    def _check_labels(self, labels):
        """
        Method that checks if the labels dimension is correct.

        # Arguments:
            labels: array with labels
        """
        if labels.ndim == 1:
            if self._n_targets != 1:
                raise AssertionError("Labels need to have the same number of targets described by the model "
                                     + str(self._n_targets) + ". Current labels have only 1 target.")
        elif labels.shape[1] != self._n_targets:
            raise AssertionError("Labels need to have the same number of targets described by the model " +
                                 str(self._n_targets) + ". Current labels have " + str(labels.shape[1]) + " targets.")

    @staticmethod
    def _check_initialization(n):
        """
        Method that checks if model's initialization is correct. 
        The number of features and targets must be an integer equal or greater to one.

        # Arguments:
            n: number of features or targets
        """
        if not isinstance(n, int):
            raise AssertionError("n_features and n_targets must be a positive integer number. Provided value "
                                 + str(n) + ".")
        if n < 1:
            raise AssertionError("n_features and n_targets must be equal or greater that 1. Provided value "
                                 + str(n) + ".")
