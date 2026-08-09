import numpy as np
from typing import Union, List

def ReLu(input_array: Union[np.ndarray, List, float, int], inplace: bool = False) -> np.ndarray:
    """
    Applies the ReLu function on a given numerical input.
    
    ReLu is defined as: max(x, 0)

    Args:
        input_array (np.ndarray): The array (or array like object) which the function will be applied to.
        inplace (bool): If True, modifies the input array directly in memory. 
                        If False, returns a newly allocated array. Defaults to False.

    Returns:
        output_array (np.ndarray): The input after having the ReLu function applied, now as a ndarray matching
                                   the precision of the input.
    
    Raises:
        TypeError: If the input cannot be safely converted into a numeric NumPy array.
        ValueError: If the input array is empty.

    Notes:
        Modifying inplace will not work for integers, as we do not want to force arrays to be int only.
    """

    # Make sure input is/can be a Numpy array and elements are numeric
    try:
        float_input = np.asarray(input_array)
        if not np.issubdtype(float_input.dtype, np.number):
            raise TypeError("Array elements must be numeric.")

    except (TypeError, ValueError) as e:
        raise TypeError(f"Input must be a numeric array or array-like object. Original error: {e}")

    # Check for empty arrays
    if float_input.size == 0:
        raise ValueError("Cannot apply ReLu to an empty array.")

    if inplace:
        # Modifying inplace forces the array to keep its dtype forever. This prevents arrays from being stuck as integers.
        if not np.issubdtype(float_input.dtype, np.floating):
            raise ValueError("In-place operations are only supported on floating-point arrays.")
        return np.maximum(float_input, 0, out=float_input)
    else:
        return np.maximum(float_input, 0)


def Softmax(input_array: np.ndarray, logit_axis: int = -1) -> np.ndarray:
    """
    Applies the Softmax function to all vectors along a specified axis in a given array.

    Softmax is defined as: exp(x) / sum(exp(x))

    Args:
        input_array (np.ndarray): The input array (or array like object) to process.
        logit_axis (int): The axis along which the model outputs sit. 
                          By convention this is the last axis (-1).

    Returns:
        np.ndarray: An array of the same shape, where elements along the logit_axis have been softmaxed.

    Raises:
        TypeError: If the input cannot be safely converted into a numeric NumPy array.
        ValueError: If the input array is empty or if logit_axis is out of bounds.

    """

    # Make sure input is/can be a Numpy array
    try:
        float_input = np.asarray(input_array, dtype=float)
    except (TypeError, ValueError) as e:
        raise TypeError(f"Input must be a numeric array or array-like object. Original error: {e}")

    # Check for empty arrays
    if float_input.size == 0:
        raise ValueError("Cannot apply softmax to an empty array.")

    #  Make sure specified axis exists in the array
    ndim = float_input.ndim
    if logit_axis >= ndim or logit_axis < -ndim:
        raise ValueError(f"Axis {logit_axis} is out of bounds for an array with {ndim} dimensions.")


    # Making all numbers <= 0 so that the 0 <= exp(num) <= 1, preventing overflow
    moved_input = float_input - np.max(float_input, axis=logit_axis, keepdims=True)
    exponents = np.exp(moved_input)
    return exponents / np.sum(exponents, axis=logit_axis, keepdims=True)