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

avg_epochs_durations = []
avg_epochs_number = []
histories = []

batch_sizes = [8, 16, 32, 64, 128, 256, 512]

N = 5

for batch_size in batch_sizes:
  hyperparams['batch_size'] = batch_size

  current_epochs_durations = []
  current_epochs_number = []

  for i in range(N):
    print("----- batch size", batch_size, ", repetition", i, "-----")

    early_stopping_callback = network['model_fit_callbacks'][0]
    time_counter_callback = TimeCounterCallback()
    network['model_fit_callbacks'] = [early_stopping_callback, time_counter_callback]

    model, history = train(config, hyperparams, network)

    current_epochs_durations.append(numpy.average(time_counter_callback.get_epoch_durations()))
    current_epochs_number.append(len(history.history['val_loss']))

    # save history of the 1st repetition only, to show on plot later
    if i == 0:
      histories.append(history.history['val_loss'])

  avg_epochs_durations.append(numpy.average(current_epochs_durations))
  avg_epochs_number.append(numpy.average(current_epochs_number))

  print("----------")
  print("batch size:", batch_size)
  print("avg epochs duration:", numpy.average(current_epochs_durations), "s")
  print("avg # of epochs:", numpy.average(current_epochs_number))
  print("----------")

print("batch sizes:", batch_sizes)
print("avg epochs durations:", avg_epochs_durations)
print("avg # of epochs:", avg_epochs_number)

with open('experiments_batch_size_1_output.txt', 'w') as f:
  print("batch sizes:", batch_sizes, file=f)
  print("avg epochs durations:", avg_epochs_durations, file=f)
  print("avg # of epochs:", avg_epochs_number, file=f)

# draw plot of first repetition for each batch size
for i in range(len(batch_sizes)):
  plt.plot(histories[i])

plt.title('Validation loss as batch size changes')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(["{}".format(batch_size) for batch_size in batch_sizes], loc='upper right')
plt.savefig('experiments_batch_size_1_plot.png')
plt.show()