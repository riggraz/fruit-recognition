from config import config, hyperparams, network
from nn import train, evaluate
import numpy

all_losses = []
avg_losses = []
variances = []

batch_sizes = [8, 16, 32, 64, 128, 256, 512]

for batch_size in batch_sizes:
  hyperparams['batch_size'] = batch_size

  current_losses = []
  repetitions = 10

  for i in range(repetitions):
    print("batch size ", batch_size, ", repetition", i)
    model, history = train(config, hyperparams, network)
    loss = evaluate(model, config)

    print("test 0-1 loss: ", loss)

    current_losses.append(loss)

  all_losses.append(current_losses)
  avg_losses.append(numpy.average(current_losses))
  variances.append(numpy.var(current_losses))

  print("----------")
  print("batch size:", batch_size)
  print("avg loss:", numpy.average(current_losses))
  print("variance:", numpy.var(current_losses))
  print(current_losses)
  print("----------")

print(batch_sizes)
print(avg_losses)
print(variances)
print(all_losses)