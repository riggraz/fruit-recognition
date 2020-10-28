from config import config, hyperparams, network
from nn import train, evaluate
import matplotlib.pyplot as plt

histories = []

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

for i in range(len(optimizers)):
  print("---------", "optimizer", optimizers[i], "---------")

  hyperparams['optimizer'] = optimizers[i]
  hyperparams['learning_rate'] = learning_rates[i]

  model, history = train(config, hyperparams, network)

  histories.append(history.history['val_loss'])

with open('experiments_optimizers_output.txt', 'w') as f:
  print("optimizers:", optimizers, file=f)
  print("learning rates:", learning_rates, file=f)
  print("val_loss histories:", histories, file=f)

for i in range(len(optimizers)):
  plt.plot(histories[i])

plt.title('Validation loss as optimizer changes')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(["{}".format(optimizer) for optimizer in optimizers], loc='upper right')
plt.savefig('experiments_optimizer_plot.png')
plt.show()