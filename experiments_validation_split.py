import reproducible_results
from config import config, hyperparams, network
from nn import train, evaluate

network['model_fit_callbacks'] = []
hyperparams['epochs'] = 10

histories = []
losses = []

splits = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

for validation_split in splits:
  print("Validation Split ", validation_split)

  hyperparams['validation_split'] = validation_split

  model, history = train(config, hyperparams, network)
  loss = evaluate(model, config)

  histories.append(history)
  losses.append(loss)


print("Losses: ", losses)

import matplotlib.pyplot as plt

for i in range(len(splits)):
  plt.plot(histories[i].history['val_accuracy'])

plt.title('Validation accuracy as validation split changes')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(["{}".format(split) for split in splits], loc='upper left')
plt.show()