import numpy as np
import cv2
import glob

from sklearn.utils import shuffle

# Dataset from
# https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap

USE_COMPRESSED = True

def get_data(path, learn_test_ratio=0.8, nbr_classes=9999, nbr_samples_per_class=9999):
    """ Returns number of classes, x_train, y_train, x_test, y_test tuple
    :type path: String
    :param path: path to dataset

    :type learn_test_ratio: float
    :param learn_test_ratio: ratio between number of samples for learn and number of sample for testing

    :type nbr_classes: int
    :param nbr_classes: [optional] number of classes to use

    :type nbr_samples_per_class: int
    :param nbr_samples_per_class: [optional] maximum number of samples to use

    :rtype: tuple
    """
    x = []
    y = []
    classes = set()
    i = 0
    ext='npy'
    if USE_COMPRESSED: ext = 'npz'
    for filename in glob.glob('{}/*.{}'.format(path, ext)):
        raw_images = np.load(filename)
        if USE_COMPRESSED : raw_images = raw_images['arr_0']
        print('loading ', filename)
        print(i)
        if i >= nbr_classes:
            break
        i += 1
        k = 0
        for raw in raw_images:
            raw = raw.astype(np.float32)
            raw /= 255.0
            x.append(raw.reshape((28, 28, 1)))
            #classname = filename.split('/')[1].split('.')[0].split('_')[3]
            classname = filename # Actually we don't really need this sh*t
            classes.add(classname)
            y.append(classname)
            if k >= nbr_samples_per_class:
                break
            k += 1

    # create lookup table
    lut_classes = dict()
    i = 0
    for c in classes:
        lut_classes[c] = i
        i += 1


    y_expected = []
    for elem in y:
        out = np.zeros(len(classes))
        out[lut_classes[elem]] = 1
        y_expected.append(out)

    xshuffle, yshuffle = shuffle(x, y_expected)

    datasetSize = len(xshuffle)
    partition = int(0.6*datasetSize)

    x_train, x_test = xshuffle[:partition], xshuffle[partition:]
    y_train, y_test = yshuffle[:partition], yshuffle[partition:]

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    print('Informations about the dataset')
    print('{} classes : {}'.format(len(classes), lut_classes))
    print('{} training samples / {} testing samples'.format( len(x_train), len(x_test) ))

    return len(classes), x_train, y_train, x_test, y_test
