from .gradient_descent import GradientDescent
from .sgd import SGD
from .adam import Adam
from .newton import NewtonMethod
from .als import ALS

ALGORITHMS = {
    "Gradient Descent": GradientDescent,
    "SGD": SGD,
    "Adam": Adam,
    "Newton Method": NewtonMethod,
    "ALS": ALS,
}
