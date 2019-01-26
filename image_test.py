import requests
import io
from PIL import Image
from PIL import ImageEnhance
import numpy as np
import cv2
import csv
import time
from keras.models import load_model
model = load_model('1.h5')

LETTERSTR = "123456789abcdefghijkmonpqrstuvwxyz"
success = 0
fail = 0
with open('label.csv', 'r', newline='') as csvfile:
    reader = csv.reader(csvfile)
    for i in reader:
        im = Image.open('black/{}.png'.format(i[0]))
        prediction = model.predict(np.expand_dims(np.stack([np.array(im.convert('1')) / 255.0]), axis=-1))
        answer = ''
        for predict in prediction:
            answer += LETTERSTR[np.argmax(predict[0])]
        if answer == i[1]:
            success = success + 1
        else:
            fail = fail +1
        print((float(success)/(success+fail)*100))
