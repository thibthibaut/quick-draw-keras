import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import keras
from keras.models import load_model

model = load_model('./supermodel2classes.h5')

print model.get_weights()[0].shape()

def inference(inputimg):
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
    pred =  np.argmax(prediction)
    print prediction
    if pred == 0: print 'Bicycle! (', prediction[0][0], '%)'
    else: print 'Cookie! (', prediction[0][1], '%)'

class Painter(object):
    def __init__(self, ax, img):
        self.showverts = True
        self.figure = plt.figure(1)
        self.button_pressed = False
        self.img = img
        self.brush_size = 20
        self.ax = ax
        self.color = 255

        canvas = self.figure.canvas
        canvas.mpl_connect('button_press_event', self.button_press_callback)
        canvas.mpl_connect('button_release_event', self.button_release_callback)
        canvas.mpl_connect('motion_notify_event', self.on_move)

    def button_press_callback(self, event):
        if(event.button == 1):
            self.button_pressed = True
            x = int(math.floor(event.xdata))
            y = int(math.floor(event.ydata))
            cv2.circle(self.img, (x, y), int(self.brush_size / 2), (self.color, self.color, self.color), -1)
            inference(self.img)
            #update the image

    def button_release_callback(self, event):
        self.button_pressed = False
        self.ax.images.pop()
        self.ax.imshow(self.img, interpolation='nearest', alpha=0.6)
        plt.draw()
        cv2.imwrite('test.png', self.img)

    def on_move(self, event):
        if(self.button_pressed):
            try:
                x = int(math.floor(event.xdata))
                y = int(math.floor(event.ydata))
                cv2.circle(self.img, (x, y), int(self.brush_size / 2), (self.color, self.color, self.color), -1)
            except:
                pass

def draw_demo():
    global imgMain
    # imgOver = np.zeros((800,800,3), np.uint8)
    imgMain = np.zeros((800,800,3), np.uint8)
    # imgMain = mpimg.imread('mycookie.png')
    #imgMain = np.random.uniform(0, 255, size=(500, 500))

    ax = plt.subplot(111)
    ax.imshow(imgMain, interpolation='nearest', alpha=1)
    # ax.imshow(imgOver, interpolation='nearest', alpha=0.6)

    pntr = Painter(ax, imgMain)
    plt.title('Click on the image to draw')
    plt.show()

if __name__ == '__main__':
    draw_demo()
