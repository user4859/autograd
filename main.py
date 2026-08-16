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
    
    Inputed arrays **MUST** have been Softmaxed before.

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

    return float(np.mean(losses))


class SequenceError(Exception):
    """Exception raised when custom operations are done out of a required sequence"""
    pass


class LinearLayer():
    """
    Simple fully connected linear layer.

    Attributes:
        random_seed (int | None): The NumPy random seed can be set to allow creating layers with 
                                  the same initial conditions repeatedly.
        precision (str): NumPy dtype string for inputs, parameters, and gradients.
        last_operation (str): Keeps track of last function called to ensure forward, backward and
                              updates happen in that order during runtime.
        weight_matrix (np.ndarray): Shape (input_size, output_size).
        bias_matrix (np.ndarray): Shape (output_size,).
        last_inputed_array (np.ndarray | None): Cached input from the last forward pass.
        dB (np.ndarray | None): Bias gradient computed in backward().
        dW (np.ndarray | None): Weight gradient computed in backward().
        passed_down_grad (np.ndarray | None): Gradient with respect to the input.
        paramaters (dict): A dictionary of the layers weight and bias arrays.

    Notes:
        - The layer computes the affine transform: output = input @ weight_matrix + bias_matrix.
        - `forward()` must be called before `backward()`.
        - `backward()` must be called before `update_paramaters()`.
    """

    

    def __init__(self, input_size: int, output_size: int, precision: str = 'float32', random_seed: int = None):
        if random_seed is not None:
            try:
               rng =  np.random.default_rng(random_seed)
            except (TypeError, ValueError) as exc:
                raise TypeError("Random seed must be int") from exc
        else:
            rng = np.random.default_rng()

        self.random_seed = random_seed
        self.precision = precision
        self.last_operation = None
        
        self.weight_matrix = np.array(rng.integers(-1000, 1000, size=(input_size,output_size)) / 1000).astype(self.precision)
        self.bias_matrix = np.array(rng.integers(-1000, 1000, size=(output_size)) / 1000).astype(self.precision)

        self.parramaters = {"weights" : self.weight_matrix, "biases" : self.bias_matrix}

        self.last_inputed_array = None
        self.dB = None
        self.dW = None
        self.passed_down_grad = None



    def forward(self, input_array: Union[np.ndarray, List, float, int]) -> np.ndarray:
        """
        Passes an array through the linear layer.
            X = W.X + B

        Args:
            input_array (Array-Like): The input array (or array like object) to process.
        
        Returns:
            np.ndarray: The inputted array is coppied for later use (back propgation) before being passed forward in place.

        Raises:
            TypeError: If the input cannot be safely converted into a numeric NumPy array.
            ValueError: If the input array is empty.
        """

        try:
            input_array = np.asarray(input_array, dtype=self.precision)
        except (TypeError, ValueError) as e:
            raise TypeError(f"Input must be a numeric array or array-like object. Original error: {e}")
        
        # Check for empty arrays
        if input_array.size == 0:
            raise ValueError("Cannot pass forward an empty array.")

        if input_array.ndim == 1:
                    input_array = np.array([input_array], dtype=self.precision)

        self.last_inputed_array = input_array.copy().astype(self.precision)
        input_array = np.dot(self.last_inputed_array, self.weight_matrix) + self.bias_matrix

        self.last_operation = "forward"

        return input_array


    def backwards(self, error_array: np.ndarray):
        """
        Calculates gradients for the layer parameters and the gradient to pass to the previous layer.

        Args:
            error_array (np.ndarray): The gradient of the loss with respect to the layer output.
                Expected shape is (batch_size, output_size).

        Updates:
            self.dB: Gradient of the loss with respect to the bias vector, summed over the batch.
            self.dW: Gradient of the loss with respect to the weight matrix.
            self.passed_down_grad: Gradient of the loss with respect to the layer input,
                to be passed to earlier layers during backpropagation.

        Notes:
            - `forward` must be called before `backward` so `self.last_inputed_array`
              contains the inputs used during the forward pass.
        """

        if self.last_operation != "forward":
            raise SequenceError(
                "A forward pass must have been compleated imidatley before the coresponding backwards pass."
                )

        error_array = error_array.astype(self.precision)

        self.dB = np.sum(error_array, axis=0, keepdims=True)
        self.dW = np.dot(self.last_inputed_array.T, error_array)
        self.passed_down_grad = np.dot(error_array, self.weight_matrix.T)

        self.last_operation = "backwards"


    def update_parameters(self, learning_rate: float):
        """
        Updates trainable parameters using gradients computed in backward().

        Args:
            learing_rate (float): Scalar learning rate for gradient descent.

        Notes:
            - `backward()` must be called first so `self.dW` and `self.dB` are defined.
            - `self.weight_matrix` is updated with `self.weight_matrix -= self.dW * learing_rate`.
            - `self.bias_matrix` is updated with `self.bias_matrix -= self.dB * learing_rate`.
        """

        if self.last_operation != "backwards":
                    raise SequenceError(
                        "A backwards pass must have been compleated imidatley before updating layer paramaters."
                        )

        try: 
            float(learning_rate)
        except (TypeError, ValueError) as exc:
           raise TypeError("Learning rate must be a number, preferably float") from exc
        
        self.weight_matrix -= self.dW * learning_rate
        self.weight_matrix = self.weight_matrix.astype(self.precision)
        self.bias_matrix -= self.dB * learning_rate
        self.bias_matrix = self.bias_matrix.astype(self.precision)


    def set_parramaters(self, parramaters : dict):
        """
        Sets the layer parameters to the ones stored in the input dictionary.

        Args:
            parramaters (dict): The dictionary storing the parramaters. **MUST** be in order weights, biases.

        Updates:
            self.weight_matrix: Sets self.weight_matrix to parramaters["weights"]
            self.bias_matrix: Sets self.bias_matrix to parramaters["biases"] 

        Rasies:
            TypeError: If the input is not a dictionary or the objects inside the dictonary can not safely be converted into numpy arrays.
            KeyError: If the inputed dictionary does not have the keys "weights" and "biases".
            ValueError: If the arrays in the dictionary are not the same shape as the existing parramater arrays.
        """

        if not isinstance(parramaters, dict):
            raise TypeError("Input - parramters - mnust be a dict.")

        try:
            new_weights = np.asarray(parramaters["weights"], dtype=self.precision)
            new_biases = np.asarray(parramaters["biases"], dtype=self.precision)
        except KeyError as exc:
            raise KeyError("Input - parramaters - dictionary must contain 'weights' and 'biases'") from exc
        except (TypeError, ValueError) as exc:
            raise TypeError("Could not convert provided parramaters to numpy arrays") from exc

        if new_weights.shape != self.weight_matrix.shape:
            raise ValueError(f"weights shape mismatch: expected {self.weight_matrix.shape}, got {new_weights.shape}")
        if new_biases.shape != self.bias_matrix.shape:
            raise ValueError(f"biases shape mismatch: expected {self.bias_matrix.shape}, got {new_biases.shape}")

        self.weight_matrix = new_weights
        self.bias_matrix = new_biases
        self.parramaters = {"weights": self.weight_matrix, "biases": self.bias_matrix}
