from config import config, hyperparams, network
from nn import train, evaluate

model, history = train(config, hyperparams, network)
loss = evaluate(model, config)

print("test 0-1 loss: ", loss)