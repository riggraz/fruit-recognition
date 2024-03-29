# Fruit image classification with convolutional neural networks

Project for the exam Statistical Methods for Machine Learning @ UniMI.

For additional information and experimental results see the [project report](project-report.pdf).

## Assignment

Use Tensorflow 2 to train neural networks for the classification of fruit/vegetable types based on images from this dataset. Images must be transformed from JPG to RGB pixel values and scaled down (e.g., 32x32). Use fruit/vegetable types (as opposed to variety) as labels to predict and consider only the 10 most frequent types (apple, banana, plum, pepper, cherry, grape, tomato, potato, pear, peach). Experiment with different network architectures and training parameters documenting their influence of the final predictive performance. While the training loss can be chosen freely, the reported test errors must be measured according to the zero-one loss for multiclass classification.

## Repository structure

- `prepare_dataset.py` is the script used to transform the [Fruits-360 dataset](https://github.com/Horea94/Fruit-Images-Dataset) into the dataset used in this experiment. For more information on how to use it, see the comments on the top of the file itself.
- `config.py` contains most of the neural network configuration (three dictionaries are set up: `config`, `hyperparams` and `network`). This file is supposed to contain a default configuration for the neural network. It should be imported in other files and, if needed, it is possible to override some of the default values provided therein.
- `nn.py` contains three functions: `train`, `evaluate` and `predict_and_show_errors`. The first two functions are used to train the neural network and evaluate it on the test set, respectively. The last one is used to show some wrong predictions that the model has made.
  - `train` takes `config`, `hyperparams` and `network` as parameters, and returns a tuple containing the trained `model` and the training `history`.
  - `evaluate` takes `model` and `config` as parameters, and returns the 0-1 test loss for multiclass classification.
  - `predict_and_show_errors` takes `model` and `config` as parameters, and outputs a plot with 9 wrongly predicted images.
- `main.py` is the simplest example showing how to train and evaluate a neural network using the `config.py` and `nn.py` files.
- `experiments_*.py` are scripts that perform experiments on a certain configuration, hyperparameter or topology. They usually define an array of values for an hyperparameter, then train the network for each of these values and plot the results.
- `reproducible_results.py` is a script that, if imported in another script in which a neural network will be trained, guarantees that if the neural network will be trained N times with the same exact parameters, it will always return the same exact model and results (the learning algorithm becomes deterministic rather than stochastic). It does so by fixing all possible python/tensorflow seeds. However, this file has not been used in the report.
