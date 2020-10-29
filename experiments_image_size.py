from config import config, hyperparams, network
from nn import train, evaluate
from tensorflow.keras import callbacks, layers
import numpy
import time
import matplotlib.pyplot as plt

class TimeCounterCallback(callbacks.Callback):
  def __init__(self):
    self.mark = 0
    self.epoch_durations = []
  def on_epoch_begin(self, epoch, logs=None):
    self.mark = time.time()
  def on_epoch_end(self, epoch, logs=None):
    if (epoch != 0): # do not count 1st epoch since it also considers shuffle buffer filling time
      self.epoch_durations.append(time.time() - self.mark)
  def get_epoch_durations(self):
    return self.epoch_durations

all_losses = []
avg_losses = []
variances = []
avg_epochs_durations = []
avg_epochs_number = []
histories = []

image_sizes = [16, 32, 64]

N = 10

for image_size in image_sizes:
  config['img_size'] = (image_size, image_size)

  # re-instantiating all layers is needed because image size changed
  network['topology'] = [
    layers.Conv2D(8, (3, 3), padding='same', activation='relu'),
    layers.MaxPool2D((2, 2)),
    layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
    layers.MaxPool2D((2, 2)),
    layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    layers.MaxPool2D((2, 2)),
    layers.Dropout(0.4),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
  ]

  current_losses = []
  current_epochs_durations = []
  current_epochs_number = []

  for i in range(N):
    print("----- image size", image_size, ", repetition", i, "-----")

    early_stopping_callback = network['model_fit_callbacks'][0]
    time_counter_callback = TimeCounterCallback()
    network['model_fit_callbacks'] = [early_stopping_callback, time_counter_callback]

    model, history = train(config, hyperparams, network)
    loss = evaluate(model, config)

    print("test 0-1 loss: ", loss)

    current_losses.append(loss)
    current_epochs_durations.append(numpy.average(time_counter_callback.get_epoch_durations()))
    current_epochs_number.append(len(history.history['val_loss']))

    # save history of the 1st repetition only, to show on plot later
    if i == 0:
      histories.append(history.history['val_loss'])

  all_losses.append(current_losses)
  avg_losses.append(numpy.average(current_losses))
  variances.append(numpy.var(current_losses))
  avg_epochs_durations.append(numpy.average(current_epochs_durations))
  avg_epochs_number.append(numpy.average(current_epochs_number))

  print("----------")
  print("image size:", image_size)
  print("avg loss:", numpy.average(current_losses))
  print("variance:", numpy.var(current_losses))
  print(current_losses)
  print("avg epochs duration:", numpy.average(current_epochs_durations))
  print("avg # of epochs:", numpy.average(current_epochs_number))
  print("----------")

print("image sizes:", image_sizes)
print("avg losses:", avg_losses)
print("variances:", variances)
print("all losses:", all_losses)
print("avg epochs durations:", avg_epochs_durations)
print("avg # of epochs:", avg_epochs_number)

with open('experiments_image_size_output.txt', 'w') as f:
  print("image sizes:", image_sizes, file=f)
  print("avg losses:", avg_losses, file=f)
  print("variances:", variances, file=f)
  print("all losses:", all_losses, file=f)
  print("avg epochs durations:", avg_epochs_durations, file=f)
  print("avg # of epochs:", avg_epochs_number, file=f)

# draw plot of first repetition for each image size
for i in range(len(image_sizes)):
  plt.plot(histories[i])

plt.title('Validation loss as image size changes')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(["{}x{}".format(image_size, image_size) for image_size in image_sizes], loc='upper right')
plt.savefig('experiments_image_size_plot.png')
plt.show()