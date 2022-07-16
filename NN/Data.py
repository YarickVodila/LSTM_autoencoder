import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error

train_data = pd.read_csv('../Data_train.csv')
train_data_2 = pd.read_csv('../Data_2_train.csv')
train_y = pd.read_csv('../Class.csv')

train_data = train_data.drop(labels = ['Unnamed: 0'],axis = 1)
train_data_2 = train_data_2.drop(labels = ['Unnamed: 0'],axis = 1)
train_y = train_y.drop(labels = ['Unnamed: 0'],axis = 1)

train_data['train_y'] = train_y

#print(train_data)
#train_data = train_data[train_data['train_y']]
#print(train_data)

train_x_1 = np.array(train_data.drop(labels = ['train_y'],axis = 1))
X_train_1, y_train = train_x_1, train_data['train_y']
scaler = preprocessing.QuantileTransformer().fit(X_train_1)
X_train_1 = scaler.transform(X_train_1)
X_train_1 = np.reshape(X_train_1, (X_train_1.shape[0], 1, 240)) #Для 1 графика

train_x_2 = np.array(train_data_2)
X_train_2 = train_x_2
scaler = preprocessing.QuantileTransformer().fit(X_train_2)
X_train_2 = scaler.transform(X_train_2)
X_train_2 = np.reshape(X_train_2, (X_train_2.shape[0], 1, 240)) #Для 2 графика

print(X_train_1.shape,'\n\n', X_train_2.shape)

m_zero = tf.keras.models.load_model("../LSTM/Data_zero.h5")
m_first = tf.keras.models.load_model("../LSTM/Data_first.h5")
m_second = tf.keras.models.load_model("../LSTM/Data_second.h5")

m_zero_2 = tf.keras.models.load_model("../LSTM/Data_2_zero.h5")
m_first_2 = tf.keras.models.load_model("../LSTM/Data_2_first.h5")
m_second_2 = tf.keras.models.load_model("../LSTM/Data_2_second.h5")

def predict_models(massiv, massiv_2):
      predict_zero_1 = m_zero.predict(massiv)
      predict_first_1 = m_first.predict(massiv)
      predict_second_1 = m_second.predict(massiv)

      predict_zero_2 = m_zero_2.predict(massiv_2)
      predict_first_2 = m_first_2.predict(massiv_2)
      predict_second_2 = m_second_2.predict(massiv_2)

      foto = []
      piezo = []


      foto.append(mean_absolute_error(predict_zero_1[0][0], massiv[0][0]))
      foto.append(mean_absolute_error(predict_first_1[0][0], massiv[0][0]))
      foto.append(mean_absolute_error(predict_second_1[0][0], massiv[0][0]))

      piezo.append(mean_absolute_error(predict_zero_2[0][0], massiv_2[0][0]))
      piezo.append(mean_absolute_error(predict_first_2[0][0], massiv_2[0][0]))
      piezo.append(mean_absolute_error(predict_second_2[0][0], massiv_2[0][0]))

      class_foto = foto.index(min(foto))
      class_piezo = piezo.index(min(piezo))

      class_sred = (class_foto + class_piezo) / 2

      if float(class_sred) != int(class_sred):
          otvet = foto.index(min(foto))
      else:
          otvet = int(class_sred)

      return otvet

pred = []
y = []

#X_train_1.shape[0]
for i in range(X_train_1.shape[0]):
    print(i)
    massiv = np.array([X_train_1[i]])
    massiv_2 = np.array([X_train_2[i]])
    pred.append(predict_models(massiv,massiv_2))
    y.append(y_train[i])

print('Метрика acuracy_score: ', accuracy_score(y, pred))