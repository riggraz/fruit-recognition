from config import config, hyperparams, network
from nn import train, evaluate, predict_and_show_errors

model, history = train(config, hyperparams, network)
loss = evaluate(model, config)
print("test 0-1 loss: ", loss)

# predict_and_show_errors(model, config)