from config import config, hyperparams, network
from nn import train, evaluate
from tensorflow.keras import layers
import numpy

all_losses = []
avg_losses = []
variances = []

dropout = 0.4
topologies = [
  [
    layers.Conv2D(8, (3, 3), padding='same', activation='relu'),
    layers.MaxPool2D((2, 2)),
    layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
    layers.MaxPool2D((2, 2)),
    layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    layers.MaxPool2D((2, 2)),
    layers.Dropout(dropout),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
  ],
  [
    layers.Conv2D(8, (3, 3), padding='same', activation='relu'),
    layers.MaxPool2D((2, 2)),
    layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
    layers.MaxPool2D((2, 2)),
    layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    layers.MaxPool2D((2, 2)),
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.MaxPool2D((2, 2)),
    layers.Dropout(dropout),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
  ],
  [
    layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    layers.MaxPool2D((2, 2)),
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.MaxPool2D((2, 2)),
    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.MaxPool2D((2, 2)),
    layers.Dropout(dropout),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
  ],
  [
    layers.Conv2D(8, (3, 3), padding='same', activation='relu'),
    layers.Conv2D(8, (3, 3), strides=(2, 2), padding='same', activation='relu'),
    layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
    layers.Conv2D(16, (3, 3), strides=(2, 2), padding='same', activation='relu'),
    layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu'),
    layers.Dropout(dropout),
    layers.Flatten(),
  ],
  [
    layers.Conv2D(8, (3, 3), padding='same', activation='relu'),
    layers.MaxPool2D((2, 2)),
    layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
    layers.MaxPool2D((2, 2)),
    layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    layers.MaxPool2D((2, 2)),
    layers.Dropout(dropout),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
  ],
]

data_augmentation = [
  False,
  False,
  False,
  False,
  [
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.5),
    layers.experimental.preprocessing.RandomContrast(1.0),
  ],
]

N = 1

for i in range(len(topologies)):
  
  if data_augmentation[i] == False:
    network['data_augmentation'] = None
  else:
    network['data_augmentation'] = data_augmentation[i]
  
  network['topology'] = topologies[i]

  current_losses = []

  for j in range(N):
    print("topology ", i, ", repetition", j)
    model, history = train(config, hyperparams, network)
    loss = evaluate(model, config)

    print("test 0-1 loss: ", loss)

    current_losses.append(loss)

  all_losses.append(current_losses)
  avg_losses.append(numpy.average(current_losses))
  variances.append(numpy.var(current_losses))

  print("----------")
  print("topology:", i)
  print("avg loss:", numpy.average(current_losses))
  print("variance:", numpy.var(current_losses))
  print(current_losses)
  print("----------")

print(topologies)
print(avg_losses)
print(variances)
print(all_losses)

with open('experiments_network_topology_output.txt', 'w') as f:
  print("topologies:", topologies, file=f)
  print("avg losses:", avg_losses, file=f)
  print("variances:", variances, file=f)
  print("all losses:", all_losses, file=f)