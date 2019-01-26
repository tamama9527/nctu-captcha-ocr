from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from PIL import Image
import numpy as np
import csv
import os

LETTERSTR = "123456789abcdefghijkmonpqrstuvwxyz"


def toonehot(text):
    labellist = []
    for letter in text:
        onehot = [0 for _ in range(34)]
        num = LETTERSTR.find(letter)
        onehot[num] = 1
        labellist.append(onehot)
    return labellist


# Create CNN Model
print("Creating CNN model...")
din = Input((30, 90,1))
out = din
out = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(out)
out = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(out)
out = BatchNormalization()(out)
out = MaxPooling2D(pool_size=(2, 2))(out)
out = Dropout(0.5)(out)
out = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(out)
out = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(out)
out = BatchNormalization()(out)
out = MaxPooling2D(pool_size=(2, 2))(out)
out = Dropout(0.5)(out)
out = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(out)
out = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(out)
out = BatchNormalization()(out)
out = MaxPooling2D(pool_size=(2, 2))(out)
out = Dropout(0.5)(out)
out = Conv2D(filters=256, kernel_size=(3, 3), activation='relu',data_format = 'channels_first')(out)
out = BatchNormalization()(out)
out = MaxPooling2D(pool_size=(2, 2))(out)
out = Flatten()(out)
out = Dropout(0.5)(out)
out = [Dense(34, name='digit1', activation='softmax')(out),\
    Dense(34, name='digit2', activation='softmax')(out),\
    Dense(34, name='digit3', activation='softmax')(out),\
    Dense(34, name='digit4', activation='softmax')(out),\
    Dense(34, name='digit5', activation='softmax')(out)]
model = Model(inputs=din, outputs=out)
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
model.summary()

print("Reading training data...")
traincsv = open('./label.csv', 'r')
train_data = np.stack([np.array(Image.open('./black/' + str(i) + ".png").convert('1'))/255.0 for i in range(1, 3506)])
train_data = np.expand_dims(train_data, axis=-1)

read_label = [toonehot(row[1]) for row in csv.reader(traincsv)]
print(len(read_label))
train_label = [[] for _ in range(5)]
for arr in read_label:
    for index in range(5):
        train_label[index].append(arr[index])
train_label = [arr for arr in np.asarray(train_label)]
print("Shape of train data:", train_data.shape)

model.fit(train_data, train_label, batch_size=150,epochs=100, verbose=2)
model.save("C:\\Users\\zeus\\Desktop\\eportal\\1.h5")