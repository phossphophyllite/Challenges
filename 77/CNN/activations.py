import numpy as np
def activation_functions_(activation):
    activations = {
        "relu": lambda x: np.maximum(0, x),
        "lrelu": lambda x: np.where(x > 0, x, 0.01 * x),
        "tanh": lambda x: np.tanh(x),
        "logistic": lambda x: 1 / (1 + np.exp(-x)),
        "softmax": lambda x: np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    }
    return activations[activation]

def activation_derivatives_(activation):
    derivatives = {
        "relu": lambda x: np.where(x > 0, 1, 0),
        "lrelu": lambda x: np.where(x > 0, 1, 0.01),
        "tanh": lambda x: 1 - np.tanh(x)**2,
        "logistic": lambda x: np.exp(-x) / (1 + np.exp(-x))**2,
        "softmax": lambda x: activation_functions_("softmax")(x) * (1 - activation_functions_("softmax")(x))
    }
    return derivatives[activation]