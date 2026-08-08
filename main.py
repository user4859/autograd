import numpy as np

def ReLu(input: np.ndarray) -> np.ndarray:
    """
    Applies the ReLu function on a given numpy array \n
    ReLu is defined as "max(x, 0)"

    Args:
        input(array): The tensor which the function will be applied to.

    Returns:
        np.array: The same array inputed, with each item having had the ReLu function appied to it.
    """
    return np.maximum(input, 0)