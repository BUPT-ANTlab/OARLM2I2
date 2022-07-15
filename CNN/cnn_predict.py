from collections.abc import Sequence
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
import pandas as pd
import shutil
import os
# Load modules
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
#from keras import regularizers
from sklearn import metrics
import csv

# Convert a Pandas dataframe to the x,y inputs that TensorFlow needs
def to_xy(df, target):
    result = []
    for x in df.columns:
        if x != target:
            result.append(x)
    # find out the type of the target column.
    target_type = df[target].dtypes
    target_type = target_type[0] if isinstance(target_type, Sequence) else target_type
    # Encode to int for classification, float otherwise. TensorFlow likes 32 bits.
    if target_type in (np.int64, np.int32):
        # Classification
        dummies = pd.get_dummies(df[target])
        return df[result].values.astype(np.float32), dummies.values.astype(np.float32)
    else:
        # Regression
        return df[result].values.astype(np.float32), df[target].values.astype(np.float32)


# Nicely formatted time string
def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)



# Encode a column to a range between normalized_low and normalized_high.
def encode_numeric_range(df, name, normalized_low=0, normalized_high=1,
                         data_low=None, data_high=None):
    if data_low is None:
        data_low = min(df[name])
        data_high = max(df[name])

    df[name] = ((df[name] - data_low) / (data_high - data_low)) \
               * (normalized_high - normalized_low) + normalized_low


# def to_sequences(seq_size, data1, data2):
#     x = []
#     y = []
#
#     for i in range(len(data1) - SEQUENCE_SIZE - 1):
#         # print(i)
#         window = data1[i:(i + SEQUENCE_SIZE)]
#         after_window = data2[i + SEQUENCE_SIZE]
#         # window = [[x] for x in window]
#         # print("{} - {}".format(window,after_window))
#         x.append(window)
#         y.append(after_window)
#
#     return np.array(x), np.array(y)


def to_sequences_cnn(former_size,pred_size, data):
    x = []
    y = []

    for i in range(len(data)-former_size-pred_size):
        # print(i)
        window = data[i:i+former_size]
        after_window = data[(i+former_size) : (i+former_size + pred_size)]
        window = [[x] for x in window]
        after_window = [[y] for y in after_window]
        # print("{} - {}".format(window,after_window))
        x.append(window)
        y.append(after_window)

    return np.array(x), np.array(y)

# df = pd.read_csv("/mnt/lizihao/CNN/ETTh1.csv")
# cols = ['bus.119.gen', 'bus.4.gen', 'bus.6.gen', 'bus.10.gen', 'bus.12.gen', 'bus.15.gen',
#         'bus.18.gen', 'bus.19.gen', 'bus.24.gen', 'bus.25.gen', 'bus.26.gen', 'bus.120.gen',
#         'bus.31.gen', 'bus.32.gen', 'bus.46.gen', 'bus.69.gen', 'bus.99.gen', 'bus.116.gen']
class cnn_predict():
    def __init__(self, load_path):
        self.N_FORMER = 84
        self.N_PRED = 28
        self.history_datas=[]
        self.cnn_model = Sequential()
        self.cnn_model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=(1, self.N_FORMER, 48),
                             padding='same'))
        self.cnn_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        self.cnn_model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
        self.cnn_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        self.cnn_model.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
        self.cnn_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        self.cnn_model.add(Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
        self.cnn_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        self.cnn_model.add(Flatten())
        self.cnn_model.add(Dense(500, activation='relu'))
        self.cnn_model.add(Dropout(0.20))
        self.cnn_model.add(Dense(self.N_PRED * 48))
        self.cnn_model.summary()
        self.cnn_model.load_weights(load_path)


    def store_his(self,data):
        self.history_datas.append(data)

    def predict(self):
        x_cnn_test=np.array(self.history_datas[-self.N_FORMER:])
        x_cnn_test = x_cnn_test.reshape(1, 1, self.N_FORMER, 48)
        cnn_model_pred = self.cnn_model.predict(x_cnn_test)
        result=np.array(cnn_model_pred)
        return(result.reshape(self.N_PRED,48))


if __name__ == '__main__':
    cnn_pre = cnn_predict()
    for i in range(84):
        cnn_pre.store_his([i%3 for _ in range(48)])
    res = cnn_pre.predict()
    print(res.shape)







#########添加裁剪数据


# df_train = df.loc[0:96074+N_FORMER+N_PRED-1].dropna()
# df_test = df.loc[96075:99075+N_FORMER+N_PRED-1].dropna()
#
#
# # df_former = df.loc[start_idx:start_idx+n_ex-1].dropna()
#
# for i in range(len(cols)):
#     encode_numeric_range(df, cols[i])
# print("DF",df)
#
# #Preparing x and y
# # SEQUENCE_SIZE = 7
# x_cnn_train,y_cnn_train = to_sequences_cnn(N_FORMER,N_PRED,df_train.values)
# x_cnn_test,y_cnn_test = to_sequences_cnn(N_FORMER,N_PRED,df_test.values)
# # x,y = to_sequences(SEQUENCE_SIZE, df.values, df_stock_close)
# #x_test,y_test = to_sequences(SEQUENCE_SIZE, df_test, close_test)
# x_cnn_train = np.squeeze(x_cnn_train)
# x_cnn_train = x_cnn_train.reshape(96074,1,N_FORMER,18)
# y_cnn_train = np.squeeze(y_cnn_train)
# y_cnn_train = y_cnn_train.reshape(96074,N_PRED*18)
# x_cnn_test = np.squeeze(x_cnn_test)
# x_cnn_test = x_cnn_test.reshape(3000,1,N_FORMER,18)
# y_cnn_test = np.squeeze(y_cnn_test)
# y_cnn_test = y_cnn_test.reshape(3000,N_PRED*18)
#
# # x_cnn = x_cnn.reshape(4384,1,7,5)
# # y_cnn = y_cnn
# #
# # x_train_cnn,x_test_cnn,y_train_cnn,y_test_cnn = train_test_split(x_cnn,y_cnn, test_size=0.3, random_state =42)
# print("Shape of x_train: {}".format(x_cnn_train.shape))
# print("Shape of x_test: {}".format(x_cnn_test.shape))
# print("Shape of y_train: {}".format(y_cnn_train.shape))
# print("Shape of y_test: {}".format(y_cnn_test.shape))
#
# cnn_model = Sequential()
# cnn_model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1),activation='relu',input_shape=(1,N_FORMER,18),padding='same'))
# cnn_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
# cnn_model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1),activation='relu',padding='same'))
# cnn_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
# cnn_model.add(Conv2D(128,kernel_size=(3, 3), strides=(1, 1),activation='relu',padding='same'))
# cnn_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
# cnn_model.add(Conv2D(256,kernel_size=(3, 3), strides=(1, 1),activation='relu',padding='same'))
# cnn_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
# cnn_model.add(Flatten())
# cnn_model.add(Dense(500, activation='relu'))
# cnn_model.add(Dropout(0.20))
# cnn_model.add(Dense(N_PRED*18))
# cnn_model.summary()
#
# # checkpointer = ModelCheckpoint(filepath="dnn/best_weights_cnn.hdf5", verbose=0, save_best_only=True) # save best model
# #
# #
# # cnn_model.compile(loss='mean_squared_error', optimizer='adam')
# # cnn_model.fit(x_cnn_train, y_cnn_train,batch_size=64,validation_data=(x_cnn_test,y_cnn_test),callbacks=[checkpointer],verbose=2,epochs=500)
# ###############TEST################
# cnn_model.load_weights('dnn/best_weights_cnn.hdf5')
#
# cnn_model_pred = cnn_model.predict(x_cnn_test)
#
# print("Shape of pred: {}".format(cnn_model_pred.shape))
# print("PRED:",cnn_model_pred)
# print("Y:",y_cnn_test)
#
# with open("mse.csv", "w", newline="") as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(["cnn_mse"])
#
# for i in range(len(cnn_model_pred)):
#     mse = metrics.mean_squared_error(y_cnn_test[i],cnn_model_pred[i])
#     with open("mse.csv", "a+", newline="") as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow([mse])
#
# score = metrics.mean_squared_error(y_cnn_test,cnn_model_pred)
#
# # print("Score (RMSE) : {}".format(score))
# # print("R2 score     :",metrics.r2_score(y_cnn_test,cnn_model_pred))
# print("MSE          :", score)

