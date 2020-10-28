# This script must be used with the Fruits-360 dataset
# See: https://github.com/Horea94/Fruit-Images-Dataset

# Run this script with Python3 or above

# Usage
# You should set the following variables according to your needs:
# src_dir, dst_dir, labels
# src_dir should contain the path of the original fruits-360 dataset
# dst_dir should contain the path of the resulting modified dataset
# (note: if dst_dir doesn't exist, the script creates it automatically)
# labels should contain a list of the labels of the new modified dataset
# (note: no whitespace allowed, just specify the fruit/vegetable type capitalized)

# What it does
# All sub-classes of the classes you specify are copied into their respective classes folder
# (e.g. "Apple Braeburn", "Apple Crimson Snow", etc., are copied into "Apple")
# The script repeat this process both for the training set and the test set automatically
# note: during the process, filenames are modified in order to avoid name collisions
# the script prepends to each filename the full label name in lower_snake_case

import os
import shutil

src_dir = './Fruits-360/'
dst_dir = './dataset2/'

labels = [
  'Apple',
  'Banana',
  'Cherry',
  'Grape',
  'Peach',
  'Pear',
  'Pepper',
  'Plum',
  'Potato',
  'Tomato'
]

for train_or_test_folder in ['Training', 'Test']:
  for label_dir in os.scandir(os.path.join(src_dir, train_or_test_folder)):
    label = label_dir.name.split(None, 1)[0]
    if (label_dir.is_dir and label in labels):
      for image_filename in os.scandir(label_dir):
        dst_filename = label_dir.name.lower().replace(" ", "_") + '_' + image_filename.name
        dst = os.path.join(dst_dir, train_or_test_folder, label, dst_filename)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(image_filename, dst)
      
      print('Copied files from {} to {}'.format(label_dir.path, os.path.join(dst_dir, train_or_test_folder, label)))
