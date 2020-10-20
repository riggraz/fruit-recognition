from config import config, hyperparams, network
from nn import train, evaluate

losses = []

for batch_size in [16, 32, 64, 128, 256]:
  print("----- batch_size = ", batch_size, " -----")
  hyperparams['batch_size'] = batch_size

  average_loss = 0.0
  repetitions = 10

  for i in range(repetitions):
    print("Repetition ", i)
    model, history = train(config, hyperparams, network)
    loss = evaluate(model, config)

    print("test 0-1 loss: ", loss)

    average_loss += loss

  average_loss /= repetitions

  losses.append(average_loss)

print(losses)