from config import config, hyperparams, network
from nn import train, evaluate
import matplotlib.pyplot as plt

histories = []

splits = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

for validation_split in splits:
  print("----- Validation Split ", validation_split, "-----")

  hyperparams['validation_split'] = validation_split

  model, history = train(config, hyperparams, network)

  histories.append(history.history['val_loss'])

with open('experiments_validation_split_output.txt', 'w') as f:
  print("validation splits:", splits, file=f)
  print("val_loss histories:", histories, file=f)

for i in range(len(splits)):
  plt.plot(histories[i])

plt.title('Validation loss as validation split changes')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(["{}".format(split) for split in splits], loc='upper right')
plt.savefig('experiments_validation_split_plot.png')
plt.show()