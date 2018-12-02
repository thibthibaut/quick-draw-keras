import numpy as np
import cv2
import glob

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

from sklearn.utils import shuffle

x = []
y = []
classes = set()

i = 0
for filename in glob.glob('dataset/*.npy'):
    raw_images = np.load(filename)
    # if i > 2:
    #     break
    # i+=1
    # k = 0
    for raw in raw_images:
        raw = raw.astype(np.float32)
        raw /= 255.0
        x.append(raw.reshape((28,28,1)))
        classname = filename.split('/')[1].split('.')[0].split('_')[3]
        classes.add(classname)
        y.append(classname)
        # if k > 100:
        #     break
        # k += 1

# create lookup table
lut_classes = dict()
i = 0
for c in classes:
    lut_classes[c] = i
    i+=1


y_expected = []
for elem in y:
    out = np.zeros(len(classes))
    out[lut_classes[elem]] = 1
    y_expected.append(out)

# Create the model
model = Sequential()
model.add(Conv2D(30, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28,28,1)))

model.add(Conv2D(50, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(120, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(classes), activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
metrics=['accuracy'])

# suffle the dataset

xshuffle, yshuffle = shuffle(x, y_expected)

datasetSize = len(xshuffle)
partition = int(0.5*datasetSize)

x_train, x_test = xshuffle[:partition], xshuffle[partition:]
y_train, y_test = yshuffle[:partition], yshuffle[partition:]

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)


model.fit(x_train, y_train,
          batch_size=128,
          epochs=10,
          verbose=1,
validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save('supermodel.h5')
