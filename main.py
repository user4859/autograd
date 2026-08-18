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


def backwards_ReLU(input_array: Union[np.ndarray, List, float, int]) -> np.ndarray:
    """
    Computes the derivative of the ReLU function element-wise.

    The derivative is 1 where x > 0 and 0 where x <= 0.

    Args:
        input_array (Array-Like): Numeric input values.

    Returns:
        np.ndarray: An array containing the ReLU derivative for each input.

    Raises:
        TypeError: If the input cannot be converted to a numeric NumPy array.
        ValueError: If the input array is empty.
    """

    # Make sure input is/can be a Numpy array and elements are numeric
    try:
        input_array = np.asarray(input_array)
        if not np.issubdtype(input_array.dtype, np.number):
            raise TypeError("Array elements must be numeric.")
    
    except (TypeError, ValueError) as e:
        raise TypeError(f"Input must be a numeric array or array-like object. Original error: {e}")

    # Check for empty arrays
    if input_array.size == 0:
        raise ValueError("Cannot apply backwards_ReLU to an empty array.")

    precision = input_array.dtype

    return (input_array > 0).astype(precision)


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

    Inputed arrays **MUST** be a probability distribution summing to 1.
    The loss is calculated as: -sum(target * log(input))
    Values smaller than a small epsilon (1e-10) are clipped to avoid log(0).

    Args:
        input_array (Array-Like): The input array (or array like object) to process.
        target_array (Array-Like): The array (or array like object) which the input will 
                                   be compared against.
        logit_axis (int): The axis along which the model's outputs sit. 
                          By convention this is the last axis (-1).

    Returns:
        float: Mean cross-entropy loss allong the logit_axis.

    Raises:
        TypeError: If either input cannot be converted to numeric arrays.
        ValueError: If shapes do not match, either array is empty, either array does not 
                    sum to 1 along the logit axis, any value is negative, or the axis is invalid.
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

    if np.any(input_array < 0) or np.any(target_array < 0):
        raise ValueError("All inputed values must be non-negative.")

    # Check arrays are acutaly probability distributions - sum to 1
    if not (
        np.allclose(np.sum(input_array, axis=logit_axis), 1.0)
        and np.allclose(np.sum(target_array, axis=logit_axis), 1.0)
    ):
        raise ValueError("Both arrays must sum to 1 along the logit axis.")

    eps = 1e-10
    clipped = np.clip(input_array, eps, None)
    losses = -np.sum(target_array * np.log(clipped), axis=logit_axis)

    return float(np.mean(losses))


derivatives = {"ReLU" : backwards_ReLU}


class SequenceError(Exception):
    """
    Exception raised when model operations are performed out of order.

    A forward pass must occur before a backward pass, and a backward pass
    must occur before parameters are updated.
    """
    pass


class LinearLayer():
    """
    A fully connected linear layer.

    The layer computes:

        output = input @ weight_matrix + bias_matrix

    Attributes:
        random_seed (int | None): Seed used to initialise the random generator.
        precision (str): NumPy dtype used for parameters, inputs, and gradients.
        last_operation (str | None): Last completed operation.
        weight_matrix (np.ndarray): Weight values with shape
                                    (input_size, output_size).
        bias_matrix (np.ndarray): Bias values with shape (output_size,).
        last_inputed_array (np.ndarray | None): Input saved during forward().
        last_outputed_array (np.ndarray | None): Output produced during forward().
        gradients (dict): Weight and bias gradients from backwards().
        parramaters (dict): Current layer weights and biases.
        passed_down_grad (np.ndarray | None): Gradient passed to the preceding layer.

    Notes:
        forward() must be called before backwards().
        backwards() must be called before update_parameters().
    """

    def __init__(self, input_size: int, output_size: int, precision: str = 'float32', random_seed: int = None, initialisation_function : str = None):
        """
        Creates and initialises a fully connected layer.

        Args:
            input_size (int): Number of input features.
            output_size (int): Number of output features.
            precision (str): NumPy dtype used by the layer.
            random_seed (int | None): Optional seed for reproducible
                                       initialisation.
            initialisation_function (str | None): Initialisation method.
                                                  "He" selects He initialisation;
                                                  other values use uniform values
                                                  between -1 and 1.

        Raises:
            TypeError: If random_seed is invalid.
        """

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

        if initialisation_function == "He":
            self.weight_matrix = np.array(rng.normal(0, np.sqrt(2/input_size), size=(input_size, output_size))).astype(self.precision)
        else:
            self.weight_matrix = np.array(rng.integers(-1000, 1000, size=(input_size,output_size)) / 1000).astype(self.precision)
        self.bias_matrix = np.array(rng.integers(-1000, 1000, size=(output_size)) / 1000).astype(self.precision)

        self.parramaters = {"weights" : self.weight_matrix, "biases" : self.bias_matrix}
        self.gradients = {"dW" : None, "dB" : None}

        self.last_inputed_array = None
        self.last_outputed_array = None
        self.passed_down_grad = None


    def forward(self, input_array: Union[np.ndarray, List, float, int]) -> np.ndarray:
        """
        Computes the forward pass through the fully conected layer.

        One-dimensional inputs are treated as a batch containing one sample
        - making them 2 dimensional.
        The input is cached for use during backpropagation.

        Args:
            input_array (Array-Like): Numeric input with shape
                                      (batch_size, input_size), or a single
                                      input vector with shape (input_size,).

        Returns:
            np.ndarray: Layer output with shape (batch_size, output_size).

        Raises:
            TypeError: If the input cannot be converted to the layer precision.
            ValueError: If the input is empty.
        """

        try:
            input_array = np.asarray(input_array, dtype=self.precision)
        except (TypeError, ValueError) as e:
            raise TypeError(f"Input must be a numeric array or array-like object. Original error: {e}")
        
        # Check for empty arrays
        if input_array.size == 0:
            raise ValueError("Can not pass forward an empty array.")

        if input_array.ndim == 1:
                    input_array = np.array([input_array], dtype=self.precision)

        self.last_inputed_array = input_array.copy().astype(self.precision)

        # Actual forward pass
        self.last_outputed_array = np.dot(self.last_inputed_array, self.weight_matrix) + self.bias_matrix

        self.last_operation = "forward"

        return self.last_outputed_array


    def backwards(self, error_array: np.ndarray):
        """
        Calculates parameter gradients and the gradient with respect to the layer input.

        Args:
            error_array (np.ndarray): Gradient of the loss with respect to
                                     the layer output, with shape
                                     (batch_size, output_size).

        Updates:
            gradients["dB"]: Bias gradient summed across the batch.
            gradients["dW"]: Weight gradient.
            passed_down_grad: Gradient with respect to the layer input.

        Raises:
            SequenceError: If forward() was not called immediately before
                           backwards().
        """

        if self.last_operation != "forward":
            raise SequenceError(
                "A forward pass must have been compleated imidatley before the a backwards pass."
                )

        error_array = error_array.astype(self.precision)

        self.gradients["dB"] = np.sum(error_array, axis=0)
        self.gradients["dW"] = np.dot(self.last_inputed_array.T, error_array)
        self.passed_down_grad = np.dot(error_array, self.weight_matrix.T)

        self.last_operation = "backwards"


    def update_parameters(self, learning_rate: float):
        """
        Updates the weights and biases using gradients computed in backwards().

        Parameters are updated according to:

            parameter = parameter - learning_rate * gradient

        Args:
            learning_rate (float): Finite scalar learning rate.

        Raises:
            TypeError: If learning_rate is not numeric.
            SequenceError: If backwards() was not called immediately before
                           update_parramaters().
        """

        if self.last_operation != "backwards":
                    raise SequenceError(
                        "A backwards pass must have been compleated imidatley before updating layer paramaters."
                        )

        try: 
            float(learning_rate)
        except (TypeError, ValueError) as exc:
           raise TypeError("Learning rate must be a number, preferably float") from exc
        
        self.weight_matrix -= self.gradients["dW"] * learning_rate
        self.weight_matrix = self.weight_matrix.astype(self.precision)
        self.bias_matrix -= self.gradients["dB"] * learning_rate
        self.bias_matrix = self.bias_matrix.astype(self.precision)

        self.parramaters["weights"] = self.weight_matrix
        self.parramaters["biases"] = self.bias_matrix


    def set_parramaters(self, parramaters : dict):
        """
        Replaces the layer's weights and biases.

        Args:
            parramaters (dict): Dictionary containing "weights" and "biases".

        Raises:
            TypeError: If the argument is not a dictionary or its values
                       cannot be converted to NumPy arrays.
            KeyError: If "weights" or "biases" is missing.
            ValueError: If either parameter has an incompatible shape.
        """

        if not isinstance(parramaters, dict):
            raise TypeError(f"Input - {parramaters} - must be a dict.")

        try:
            new_weights = np.asarray(parramaters["weights"], dtype=self.precision)
            new_biases = np.asarray(parramaters["biases"], dtype=self.precision)
        except KeyError as exc:
            raise KeyError(f"Input - {parramaters} - dictionary must contain 'weights' and 'biases'") from exc
        except (TypeError, ValueError) as exc:
            raise TypeError("Could not convert provided parramaters to numpy arrays") from exc

        if new_weights.shape != self.weight_matrix.shape:
            raise ValueError(f"weights shape mismatch: expected {self.weight_matrix.shape}, got {new_weights.shape}")
        if new_biases.shape != self.bias_matrix.shape:
            raise ValueError(f"biases shape mismatch: expected {self.bias_matrix.shape}, got {new_biases.shape}")

        self.weight_matrix = new_weights.copy()
        self.bias_matrix = new_biases.copy()
        self.parramaters = {"weights": self.weight_matrix, "biases": self.bias_matrix}


class Model():
    """
    A sequential neural network composed of fully conected linear layers, 
    activation functions and a normalisation step.

    The model contains an input layer, zero or more hidden layers, an output
    layer, and a normalisation function. Hidden layers use the selected
    activation function.

    Attributes:
        input_size (int): Number of input features.
        hidden_size (int): Number of features in hidden layers.
        output_size (int): Number of model outputs.
        number_of_layers (int): Number of fully conected linear layers.
        precision (str): NumPy dtype used throughout the model.
        random_seed (int | None): Optional initialisation seed.
        initialisation_function (str | None): Parameter initialisation method.
        activation_function (callable): Function used after hidden layers.
        normalisation_function (callable): Function used after the output layer.
        linear_layers (list): LinearLayer objects in execution order.
        modules (list): Ordered model operations.
        gradients (dict): Gradients grouped by layer.
        parramaters (dict): Parameters grouped by layer.
        last_operation (str | None): Most recently completed model operation.

    Notes:
        If number_of_layers = 1, then the input layer is the output layer,
        and only the normalisation function is used.
        The current backwards() implementation supports Softmax followed by
        CrossEntropyLoss.
    """

    def __init__(self,
                  input_size : int, hidden_size : int, output_size : int,
                  number_of_layers : int, 
                  activation_function: callable, normalisation_function: callable,
                  precision : str = 'float32', random_seed : int = None,
                  initialisation_function : str = None):
        """
        Constructs the model.

        Args:
            input_size (int): Number of input features.
            hidden_size (int): Number of features in hidden layers.
            output_size (int): Number of output features.
            number_of_layers (int): Number of linear layers.
            activation_function (callable): Activation applied after hidden
                                            linear layers.
            normalisation_function (callable): Function applied after the
                                               final linear layer.
            precision (str): NumPy dtype used by the model.
            random_seed (int | None): Optional seed for reproducible weights.
            initialisation_function (str | None): Weight initialisation method.
        """

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.number_of_layers = number_of_layers

        self.precision = precision
        self.random_seed = random_seed
        self.initialisation_function = initialisation_function

        self.activation_function = activation_function
        self.normalisation_function = normalisation_function

        self.activation_function_derivative = derivatives.get(
            getattr(self.activation_function, "__name__", None)
        )
        self.normalisation_function_derivative = derivatives.get(
            getattr(self.normalisation_function, "__name__", None)
        )


        layers = []
        self.linear_layers = []

        if self.number_of_layers == 1:
            layer = LinearLayer(self.input_size, self.output_size,
                                    self.precision, self.random_seed,
                                    self.initialisation_function)
            layers.append([layer])
            self.linear_layers.append(layer)
        else:
            input_layer = LinearLayer(self.input_size, self.hidden_size,
                                          self.precision, self.random_seed,
                                          self.initialisation_function)
            layers.append([input_layer, self.activation_function])
            self.linear_layers.append(input_layer)

            for _ in range(max(0, self.number_of_layers - 2)):
                hidden_layer = LinearLayer(self.hidden_size, self.hidden_size,
                                              self.precision, self.random_seed,
                                              self.initialisation_function)
                layers.append([hidden_layer, self.activation_function])
                self.linear_layers.append(hidden_layer)

            output_layer = LinearLayer(self.hidden_size, self.output_size,
                                          self.precision, self.random_seed,
                                          self.initialisation_function)
            layers.append([output_layer])
            self.linear_layers.append(output_layer)

        layers.append([self.normalisation_function])

        self.modules = layers
        # self.gradients serves no functional purposse, but make it easier to acces all the differnt layer's gradients at once.
        self.gradients = {f"Layer {i}": {} for i in range(1, self.number_of_layers + 1)}

        self.parramaters = {}
        for i, layer in enumerate(self.linear_layers, start=1):
            self.parramaters[f"Layer {i}"] = layer.parramaters

        self.last_operation = None


    def forward(self, x: Union[np.ndarray, List, float, int]) -> np.ndarray:
        """
        Executes a forward pass through every model module in order.

        Args:
            x (Array-Like): Numeric model input.

        Returns:
            np.ndarray: The model's final output.

        Raises:
            TypeError: If a module is neither callable nor an object with a
                       callable forward() method.
        """

        for module in self.modules:
            for obj in module:
                if hasattr(obj, "forward") and callable(obj.forward):
                    x = obj.forward(x)
                elif callable(obj):
                    x = obj(x)
                else:
                    raise TypeError(f"Module - {type(obj).__name__} - does not have a 'forward' function, or is not callable and so is not supported")
        self.last_operation = "forward"
        return x


    def backwards(self, output : np.ndarray, target : np.ndarray, loss_function):
        """
        Backpropagates the loss gradient through all linear layers.

        For Softmax followed by mean CrossEntropyLoss, the output gradient is:

            (output - target) / batch_size

        Gradients are calculated in reverse layer order and stored in the model
        and individual layers.

        Args:
            output (np.ndarray): Model output from the preceding forward pass.
            target (np.ndarray): Expected target values.
            loss_function (callable): Loss function used for training.

        Raises:
            SequenceError: If forward() was not called immediately beforehand.
            TypeError: If inputs are non-numeric or the configured derivative
                       combination is unsupported.
            ValueError: If either input is empty or their shapes differ.
        """

        if self.last_operation != "forward":
            raise SequenceError("Must compleate a forwards pass imidatley before a backwards pass.")

        try:
            output = np.asarray(output, dtype=self.precision)
            target = np.asarray(target, dtype=self.precision)
        except (TypeError, ValueError) as e:
            raise TypeError(f"The inputed arrays - {output} and {target} must be a numeric array or array-like object. Original error: {e}")

        if output.size == 0 or target.size == 0:
            raise ValueError("At least one input is empty.")

        if output.ndim == 1:
            output = np.array([output], dtype=self.precision)

        if target.ndim == 1:
            target = np.array([target], dtype=self.precision)

        if output.shape != target.shape:
            raise ValueError(f"Output shape {output.shape} does not match target shape {target.shape}.")

        # For Softmax + mean CrossEntropyLoss, the derivative with respect to logits is
        # (softmax_output - target) / batch_size. This matches the loss scaling used in
        # CrossEntropyLoss.
        if self.normalisation_function is Softmax and loss_function is CrossEntropyLoss:
            passed_down_grad = (output - target) / output.shape[0]
        else:
            raise TypeError(
                f"This model's normalisation function - {self.normalisation_function} - currently has no programed back propogtion rules."
            )

        for layer_index in range(self.number_of_layers, 0, -1):
            module = self.modules[layer_index - 1]
            layer = module[0]

            if len(module) > 1 and self.activation_function is ReLU:
                passed_down_grad = passed_down_grad * backwards_ReLU(layer.last_outputed_array)

            layer.backwards(passed_down_grad)

            self.gradients[f"Layer {layer_index}"] = {
                "dW": layer.gradients["dW"].copy(),
                "dB": layer.gradients["dB"].copy(),
            }
            passed_down_grad = layer.passed_down_grad

        self.last_operation = "backwards"


    def update_parramaters(self, learning_rate : float):
        """
        Updates every linear layer using its computed gradients.

        Args:
            learning_rate (float): Finite scalar learning rate used for
                                   gradient descent.

        Raises:
            TypeError: If learning_rate is not numeric.
            ValueError: If learning_rate is not finite.
            SequenceError: If backwards() was not called immediately beforehand.
        """
        
        if self.last_operation != "backwards":
            raise SequenceError("Must compleate a backwards pass imidatley before updating a models parramaters.")

        try: 
            float(abs(learning_rate))
        except (TypeError, ValueError) as exc:
            raise TypeError("Learning rate must be a number, preferably float") from exc

        if not np.isfinite(learning_rate):
            raise ValueError(f"Learning rate must be finite, currently it is - {learning_rate}")

        for layer_index in range(self.number_of_layers, 0, -1):
            layer = self.modules[layer_index - 1][0]
            layer.update_parameters(learning_rate)
            self.parramaters[f"Layer {layer_index}"] = layer.parramaters

        self.last_operation = "update"
        

    def set_parramaters(self, parramaters : dict):
        """
        Replaces all model layer parameters.

        Args:
            parramaters (dict): Dictionary containing one entry for each layer,
                                named "Layer 1", "Layer 2", and so on. Each
                                entry must contain "weights" and "biases".

        Raises:
            TypeError: If parramaters is not a dictionary.
            KeyError: If the layer keys do not match or required parameters
                      are missing.
            ValueError: If the number of parameter groups differs from the
                        current model.
        """

        if not isinstance(parramaters, dict):
            raise TypeError(f"Input - {parramaters} - must be a dict.")

        if set(parramaters.keys()) != set(self.parramaters.keys()):
            raise KeyError(f"Input - {parramaters} - keys do not match those in self.parramaters.")

        if len(self.parramaters) != len(parramaters):
            raise ValueError(f"Inputed dictionary has a different number of elements than the existing dictionary.")

        
        modules_copy = self.modules
        layer_index = 1
        for obj in modules_copy:
            if isinstance(obj, LinearLayer):
                key = f"Layer {layer_index}"
                if key not in parramaters:
                    raise KeyError(f"Missing parameter set for {key}.")
                obj.set_parramaters(parramaters[key])
                layer_index += 1

        self.modules = modules_copy
        self.parramaters = parramaters
