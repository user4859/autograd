import numpy as np
from typing import Union, List

def ReLU(input_array: Union[np.ndarray, List, float, int], inplace: bool = False) -> np.ndarray:
    """
    Applies the ReLU function on a given numerical input.
    
    ReLU is defined as: max(x, 0)

    Args:
        input_array (Array-Like): The array (or array like object) which the function will be applied to.
        inplace (bool): If True, modifies the input array directly in memory. 
                        If False, returns a newly allocated array. Defaults to False.

    Returns:
        output_array (np.ndarray): The input after having the ReLU function applied, now as a ndarray matching
                                   the precision of the input.
    
    Raises:
        TypeError: If the input cannot be safely converted into a numeric NumPy array.
        ValueError: If the input array is empty.

    Notes:
        Modifying inplace will not work for integers, as we do not want to force arrays to be int only.
    """

    # Make sure input is/can be a Numpy array and elements are numeric
    try:
        float_array = np.asarray(input_array)
        if not np.issubdtype(float_array.dtype, np.number):
            raise TypeError("Array elements must be numeric.")

    except (TypeError, ValueError) as e:
        raise TypeError(f"Input must be a numeric array or array-like object. Original error: {e}")

    # Check for empty arrays
    if float_array.size == 0:
        raise ValueError("Cannot apply ReLU to an empty array.")

    if inplace:
        # Modifying inplace forces the array to keep its dtype forever. This prevents arrays from being stuck as integers.
        if not np.issubdtype(float_array.dtype, np.floating):
            raise ValueError("In-place operations are only supported on floating-point arrays.")
        return np.maximum(float_array, 0, out=float_array)
    else:
        return np.maximum(float_array, 0)


def Softmax(input_array: Union[np.ndarray, List, float, int], logit_axis: int = -1) -> np.ndarray:
    """
    Applies the Softmax function to all vectors along a specified axis in a given array.

    Softmax is defined as: exp(x) / sum(exp(x))

    Args:
        input_array (Array-Like): The input array (or array like object) to process.
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
        float_array = np.asarray(input_array, dtype=float)
    except (TypeError, ValueError) as e:
        raise TypeError(f"Input must be a numeric array or array-like object. Original error: {e}")

    # Check for empty arrays
    if float_array.size == 0:
        raise ValueError("Cannot apply softmax to an empty array.")

    #  Make sure specified axis exists in the array
    ndim = float_array.ndim
    if logit_axis >= ndim or logit_axis < -ndim:
        raise ValueError(f"Axis {logit_axis} is out of bounds for an array with {ndim} dimensions.")


    # Making all numbers <= 0 so that the 0 <= exp(num) <= 1, preventing overflow
    moved_input = float_array - np.max(float_array, axis=logit_axis, keepdims=True)
    exponents = np.exp(moved_input)
    return exponents / np.sum(exponents, axis=logit_axis, keepdims=True)

def CrossEntropyLoss(input_array: Union[np.ndarray, List, float, int],
                     target_array: Union[np.ndarray, List, float, int],
                     logit_axis: int = -1) -> float:
    """
    Computes cross-entropy loss between an input array and a target array.

    Args:
        input_array (Array-Like): The input array (or array like object) to process.
        target_array (Array-Like): The array (or array like object) which the input will 
                                   be compared against.
        logit_axis (int): The axis along which the model outputs sit. 
                          By convention this is the last axis (-1).

    Returns:
        float: Scalar cross-entropy loss as a float.

    Raises:
        TypeError: If inputs cannot be converted to numeric arrays.
        ValueError: If shapes do not match, the array is empty, values are negative,
                    or the axis is invalid.
    """
    
    try:
        input_array = np.asarray(input_array, dtype=float)
        target_array = np.asarray(target_array, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"Input and target arrays must be numeric array-like objects. Original error: {exc}"
        ) from exc

    if input_array.shape != target_array.shape:
        raise ValueError("Input and target arrays must have the same shape.")

    if input_array.size == 0:
        raise ValueError("Arrays have no values.")

    ndim = input_array.ndim
    if logit_axis >= ndim or logit_axis < -ndim:
        raise ValueError(f"Axis {logit_axis} is out of bounds for an array with {ndim} dimensions.")

    if np.any(input_array < 0):
        raise ValueError("Input values must be non-negative.")

    eps = 1e-10
    clipped = np.clip(input_array, eps, None)
    losses = -np.sum(target_array * np.log(clipped), axis=logit_axis)
    print(losses)

    return float(np.mean(losses))