from tensorflow.keras import layers, callbacks, models

config = {}

config['train_ds_path'] = './dataset/Training'
config['test_ds_path'] = './dataset/Test'

config['img_size'] = (32, 32)
config['img_color_mode'] = 'rgb'

hyperparams = {}

hyperparams['validation_split'] = 0.2
hyperparams['batch_size'] = 32
hyperparams['epochs'] = 1000
hyperparams['optimizer'] = 'adam'
hyperparams['learning_rate'] = 0.001

network = {}

dropout = 0.4

network['topology'] = [
  layers.Conv2D(8, (3, 3), padding='same', activation='relu'),
  layers.MaxPool2D((2, 2)),
  layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
  layers.MaxPool2D((2, 2)),
  layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
  layers.MaxPool2D((2, 2)),
  layers.Dropout(dropout),
  layers.Flatten(),
  layers.Dense(256, activation='relu'),
]

training_early_stopping = callbacks.EarlyStopping(
  monitor='val_loss',
  patience=10,
  verbose=2,
  restore_best_weights=True
)

network['model_fit_callbacks'] = [ training_early_stopping ]

# network['data_augmentation'] = models.Sequential([
#   layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
#   layers.experimental.preprocessing.RandomRotation(0.5),
#   layers.experimental.preprocessing.RandomContrast(1.0),
# ])

network['data_augmentation'] = None