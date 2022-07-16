#Импорт библиотек
from keras.models import Model
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


#чтение и обработка данных
train_data = pd.read_csv('../Data_2.csv')
train_y = pd.read_csv('../Class_Data_2.csv')
train_data = train_data.drop(labels = ['Unnamed: 0'],axis = 1)
train_y = train_y.drop(labels = ['Unnamed: 0'],axis = 1)
train_data['train_y'] = train_y
train_data = train_data[train_data['train_y']==2] #2 - класс, для которого обучаем модель
train_x = np.array(train_data.drop(labels = ['train_y'],axis = 1))
scaler = preprocessing.QuantileTransformer().fit(train_x)
train_x = scaler.transform(train_x)
train_x = np.reshape(train_x, (train_x.shape[0], 1, 240))
X_train, y_train = train_x, train_x


#модель lstm autoencoder
model = keras.Sequential([
  keras.layers.LSTM(256, activation='relu', input_shape=(1, 240), return_sequences=True),
  keras.layers.LSTM(128, activation='relu', return_sequences=True),
  keras.layers.LSTM(64, activation='relu', return_sequences=False),
  keras.layers.RepeatVector(1),
  keras.layers.LSTM(64, activation='relu', return_sequences=True),
  keras.layers.LSTM(128, activation='relu', return_sequences=True),
  keras.layers.LSTM(256, activation='relu', return_sequences=True),
  keras.layers.TimeDistributed(keras.layers.Dense(240))
])

#компиляция и обучение
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mae')
his = model.fit(X_train, y_train, epochs = 450, batch_size = 32, shuffle = True)

#предсказание
pred = model.predict(X_train)

#визуализация
for i in range(15):
  plt.plot(pred[i][0], 'r')
  plt.plot(X_train[i][0])
  print(np.corrcoef(pred[i], X_train[i]))
  plt.show()

#сохранение модели
model.save('Data_2_second.h5')