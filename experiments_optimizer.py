# import reproducible_results
from config import config, hyperparams, network
from nn import train, evaluate

from tensorflow.keras import optimizers

import matplotlib.pyplot as plt
import numpy

hyperparams['batch_size'] = 64

histories = []
avg_losses = []
all_losses = []
variances = []

optimizers = [
  'adam',
  'adamax',
  'sgd',
]

learning_rates = [
  0.001,
  0.001,
  0.01,
]

def calc_test_error(repetitions=10):
  for i in range(len(optimizers)):
    hyperparams['optimizer'] = optimizers[i]
    hyperparams['learning_rate'] = learning_rates[i]

    current_losses = []

    for j in range(repetitions):
      print("---------", "optimizer", optimizers[i], ", repetition", j, "---------")

      model, history = train(config, hyperparams, network)
      loss = evaluate(model, config)

      current_losses.append(loss)

    all_losses.append(current_losses)
    avg_losses.append(numpy.average(current_losses))
    variances.append(numpy.var(current_losses))

  print("optimizers:", optimizers)
  print("learning rates:", learning_rates)
  print("avg losses:", avg_losses)
  print("variances:", variances)
  print("all losses:", all_losses)

  with open('experiments_optimizer_output.txt', 'w') as f:
    print("optimizers:", optimizers, file=f)
    print("learning rates:", learning_rates, file=f)
    print("avg losses:", avg_losses, file=f)
    print("variances:", variances, file=f)
    print("all losses:", all_losses, file=f)

def plot_val_acc(epochs=100):
  tmp = (network['model_fit_callbacks'], hyperparams['epochs'])
  network['model_fit_callbacks'] = []
  hyperparams['epochs'] = epochs

  for i in range(len(optimizers)):
    print("---------", "optimizer", optimizers[i], "---------")

    hyperparams['optimizer'] = optimizers[i]
    hyperparams['learning_rate'] = learning_rates[i]

    model, history = train(config, hyperparams, network)

    histories.append(history)

  for i in range(len(optimizers)):
    plt.plot(histories[i].history['val_accuracy'])

  plt.title('Validation accuracy as optimizer changes')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(["{}".format(optimizer) for optimizer in optimizers], loc='upper left')
  plt.savefig('experiments_optimizer_plot.png')
  plt.show()

  # restore previous values
  network['model_fit_callbacks'], hyperparams['epochs'] = tmp

calc_test_error(repetitions=10)
plot_val_acc(epochs=5)