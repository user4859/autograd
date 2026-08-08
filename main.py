import numpy as np

def ReLu(input: np.ndarray) -> np.ndarray:
    """
    Applies the ReLu function on a given numpy array \n
    ReLu is defined as "max(x, 0)"

    Args:
        input(array): The array which the function will be applied to.

    Returns:
        np.array: The same array inputed, with each item having had the function appied to it.
    """

    return np.maximum(input, 0)


def Softmax(input: np.ndarray, chosen_axis: int = 1) -> np.ndarray:
    """
    Applies the Softmax function to all vectors along a specified axis in a given array \n
    Softmax is defined as "exp(vector) / sum(exp(vector))"

    Args:
        input(array): The array which the function will be applied to.
        chosen_axis(int): The axis along which the softmax is calculated (by convention this is \
        axis 1.)

    Returns:
        array: An array of the same size, with each vector having had the function appied to it.

    """

    # forcing the array to be made of floats so that 
    float_input = input.astype(float)
    # making all numbers <= 0 so that the 0 <= exp(num) <= 1, preventing overflow
    moved_input = float_input - np.max(float_input, axis=chosen_axis, keepdims=True)
    exponents = np.exp(moved_input)
    return exponents / np.sum(exponents, axis=chosen_axis, keepdims=True)