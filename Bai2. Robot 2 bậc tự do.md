## **Library**
```
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from keras.utils import np_utils
from sklearn.utils import shuffle
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle
import tensorflow as tf
import math as m
```
## **Plot Function**
```
def plot_history(history_fine):
  f1 = history_fine.history['acc']
  val_f1 = history_fine.history['val_acc']

  loss = history_fine.history['loss']
  val_loss = history_fine.history['val_loss']

  plt.figure(figsize=(8, 8))
  plt.subplot(2, 1, 1)
  plt.plot(f1, label='Acc')
  plt.plot(val_f1, label='Validation Acc')
  plt.legend(loc='lower right')
  plt.title('Accuracy')

  plt.subplot(2, 1, 2)
  plt.plot(loss, label='Loss')
  plt.plot(val_loss, label='Validation Loss')
  plt.legend(loc='upper right')
  plt.title('Loss')
  plt.xlabel('epoch')
  plt.show()

def plot_reg_history(history_fine):
  loss = history_fine.history['loss']
  val_loss = history_fine.history['val_loss']
  plt.subplot(2, 1, 2)
  plt.plot(loss, label='Loss')
  plt.plot(val_loss, label='Validation Loss')
  plt.legend(loc='upper right')
  plt.title('Loss')
  plt.xlabel('epoch')
  plt.show()
```
## **Load data**
```
from sklearn.preprocessing import StandardScaler
# Define Variables
l1 = 40
l2 = 50
x_train = []
y_train = []

# Create Data
for t1 in np.linspace(-(2 * np.pi), 2 * np.pi, 500):
  for t2 in np.linspace(-(2 * np.pi), 2 * np.pi, 500):
    x = l1*m.cos(t1) + l2*m.cos(t1+t2)
    y = l1*m.sin(t1) + l2*m.sin(t1+t2)
    x_train.append(np.array([x,y]))
    y_train.append(np.array([t1,t2]))

# Convert to array
scaler = StandardScaler()
x_train = np.array(scaler.fit_transform(x_train))
y_train = np.array(y_train)

# Shuffe
x_train, y_train = shuffle(x_train, y_train)
```
## **Create model**
```
model = Sequential()
model.add(Dense(256, activation='relu', input_shape = (2,)))
model.add(Dense(256, activation='relu'))
model.add(Dense(2, activation='linear'))

model.compile(loss='mae', optimizer =tf.optimizers.Adam(learning_rate=0.0001))

history = model.fit(x_train, y_train, batch_size = 512, epochs = 50, validation_split = 0.2)

plot_reg_history(history)

```
## **Predict**
```
# Input x =90, y = 0
test = scaler.transform(np.array([[90,0]]))
t1 = model.predict(test)[0][0]
t2 = model.predict(test)[0][1]

x = l1*m.cos(t1) + l2*m.cos(t2+t1)
y = l1*m.sin(t1) + l2*m.sin(t2+t1)

print("Input x = 90 và y = 0 là t1 = " + str(t1) + " t2 = "+ str(t2))
print("Test: ")
print("Output x = " + str(x) + " y = "+ str(y))
```
