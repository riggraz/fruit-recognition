from config import config, hyperparams, network
from nn import train, evaluate
from tensorflow.keras import layers
import matplotlib.pyplot as plt

histories = []

dropouts = [0.1, 0.2, 0.3, 0.4, 0.5]

for dropout in dropouts:
  print("----- Dropout", dropout, "-----")

  network['topology'] = [
    layers.Conv2D(8, (3, 3), padding='same', activation='relu'),
    layers.MaxPool2D((2, 2)),
    layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
    layers.MaxPool2D((2, 2)),
    layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    layers.MaxPool2D((2, 2)),
    layers.Dropout(dropout), # here is the dropout!
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
  ]

  model, history = train(config, hyperparams, network)

  histories.append(history.history['val_loss'])

with open('experiments_dropout_output.txt', 'w') as f:
  print("dropouts:", dropouts, file=f)
  print("val_loss histories:", histories, file=f)

for i in range(len(dropouts)):
  plt.plot(histories[i])

plt.title('Validation loss as dropout rate changes')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(["{}".format(dropout) for dropout in dropouts], loc='upper right')
plt.savefig('experiments_dropout_plot.png')
plt.show()