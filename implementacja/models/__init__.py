from .linear_regression import LinearRegressionModel
from .logistic_regression import LogisticRegressionModel
from .ridge_regression import RidgeRegressionModel

MODELS = {
    "Linear Regression": LinearRegressionModel,
    "Ridge Regression": RidgeRegressionModel,
    "Logistic Regression": LogisticRegressionModel,
}
