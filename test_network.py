import keras
import cv2
import numpy as np

from keras.models import load_model
model = load_model('./supermodel2classes.h5')

inputimg = cv2.imread('./mybike.png')
gray_image = cv2.cvtColor(inputimg, cv2.COLOR_BGR2GRAY)
gray_image = cv2.resize(gray_image,(28, 28))

# cv2.imshow('inputimg', gray_image)
# cv2.waitKey(0)

inputdata= np.array(gray_image)
inputdata = inputdata.astype(np.float32) / 255.0


inp = []
inputdata = inputdata.reshape((28, 28, 1))
inp.append(inputdata)
inp = np.array(inp)

prediction = model.predict(inp)

print prediction[0]











