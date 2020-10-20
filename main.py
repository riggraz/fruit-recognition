from config import config, hyperparams, network
from nn import train, evaluate

hyperparams['batch_size'] = 64

model, history = train(config, hyperparams, network)
loss = evaluate(model, config)

print("test 0-1 loss: ", loss)