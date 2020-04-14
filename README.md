# CMPT 414 Project
Gesture recognition and motion controls using just a web camera.

This project has been tested on Windows only.
By: Michael Lu

## Requred Libaries
  * numpy
  * opencv
  * pyautogui
  * Keras with TensorFlow as it's backend

## Necessary Gear
  * Web Camera
  * Blue glove

## Installation On Windows
Install python 3.7 with [anaconda](https://www.anaconda.com/distribution/)
```
conda update anaconda
conda install -c conda-forge opencv
conda install -c conda-forge pyautogui

conda install -c conda-forge tensorflow
conda install -c conda-forge keras

pip3 install pyautogui
```

# How To Run
```
python main.py
```

# File Structure
Summary of what you can find in files/folders
```
cnn.py - own implementation of a cnn
back.py - own implementation of back subsitution
main.py - hand segmentation and extraction, 
          giving hand image to cnn to predict label,
          displaying frames from web camera,
          mouse movement
cnn_keras.py - cnn using keras

cnn_kerash.h5 - weights of keras model
convo_weights.npy - weights of my implementation of my cnn
data/ - contains all pictures used to train the cnn
reports/ - contains images used in report
```
