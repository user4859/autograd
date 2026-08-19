# Autograd

A small neural-network library built from scratch using NumPy. This project was created to develop a deeper understanding of the mathematics and algorithms behind machine-learning libraries such as PyTorch.

The library includes the basic components needed to build and train a fully connected classification model, including LinearLayers, ReLU, Softmax, backpropagation and gradient descent. The project also includes a Demonstration notebook where linear regression and a MNIST classifier are created, trained and shown.

## Features

- ReLU activation and derivative
- Softmax normalisation
- Cross-entropy loss
- Fully connected linear layers
- Forward propagation
- Backpropagation for Softmax with cross-entropy loss
- Gradient-descent parameter updates
- Basic parameter saving and loading
- MNIST classification demonstration

## Project Structure

- `main.py` - The core NumPy-based library.
- `Demonstration.ipynb` - Demonstrations of the library components, linear regression, and MNIST classification.
- `build.ipynb` - Development and testing notebook used while building the project.
- `reduced_MNIST_data` - Reduced MNIST dataset used by the demonstration notebook (~1MB).
- `pretrained_model_parameters.pkl` - Saved model parameters used by the demonstration.
- `LICENSE` - GNU General Public License version 3.

## Requirements

Only Python 3 and NumPy are needed to use the main.py library, however to use Demonstration.ipynb the following are needed:

- Python 3
- NumPy
- Pandas
- Matplotlib
- Jupyter Notebook or Visual Studio Code with the Jupyter extension

Install the Python dependencies with:

```bash
python -m pip install numpy pandas matplotlib jupyter
```

## Usage

Import the library and create a model:

```python
import main

model = main.Model(
    input_size,
    hidden_size,
    output_size,
    number_of_layers,
    activation_function=main.ReLU,
    normalisation_function=main.Softmax
)
```

A basic training loop can be written as follows:

```python
for _ in range(100):
    output = model.forward(X_train)
    loss = main.CrossEntropyLoss(output, Y_train)
    model.backwards(output, Y_train, main.CrossEntropyLoss)
    model.update_parameters(0.05)
```

The complete workflow, including data preparation and visual predictions, is available in `Demonstration.ipynb`.

## Results

The MNIST demonstration trains a small fully connected neural network and evaluates it on a held-out development set. The notebook also loads a set of pretrained parameters to demonstrate how a trained model can be reused.

The model is intended for education and experimentation, not production use. Training speed and accuracy are very limited compared with established machine-learning libraries.

## Limitations and Possible Improvements

- Only fully connected linear layers are currently implemented.
- Backpropagation currently only supports the Softmax and cross-entropy combination or raw output.
- There are no advanced optimisers such as Adam or momentum-based gradient descent.
- More activation functions, initialisation methods, tests, and numerical checks could be added.

However, all of the above could be added quickly, as the hard part of creating the inital systems is done.

## Learning Resources

The following resources helped with the mathematics, algorithms, and Python implementation:

- [3Blue1Brown - Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [CodeEmporium - Backpropagation by hand](https://www.youtube.com/watch?v=12-HUfbyGso&list=WL&index=4&t=1022s)
- [Samson Zhang - Building a neural network FROM SCRATCH](https://www.youtube.com/watch?v=w8yWXqWQYmU)
- [GeeksforGeeks](https://www.geeksforgeeks.org/)

AI tools, including Gemini and ChatGPT, were used for explanations, syntax help, debugging suggestions, and research. **No** function was blindly copied or fully created by an AI tool.

## License

This project is licensed under the GNU General Public License version 3. See [LICENSE](LICENSE) for the full license text.
