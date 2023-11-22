# In the name of God
import numpy as np
from random import randrange
from scipy.ndimage import rotate
from scipy.ndimage import zoom
import scipy


def flipping(x_train, y_train):
    flip_index = np.random.choice([True, False])
    new_img = np.flip(x_train, flip_index)
    new_mask = np.flip(y_train, flip_index)

    return (new_img, new_mask)


##################rotating

def rotating(x_train, y_train):
    rot = randrange(10)
    x_train = rotate(x_train, rot, reshape=False, mode='reflect')
    y_train = rotate(y_train, rot, reshape=False, mode='reflect')
    return (x_train, y_train)


##################zooming

def zooming(x_train, y_train):
    rot = randrange(1, 2)
    x_train = zoom(x_train, rot, mode='reflect')
    y_train = zoom(y_train, rot, mode='reflect')
    return (x_train, y_train)


def train_data_generator(x_train, y_train, batch_size):
    while True:
        images = []
        masks = []
        for i in range(batch_size):
            img_index = randrange(41)

            x_image = x_train[img_index]
            y_mask = y_train[img_index]
            aug_index = np.random.choice([0, 1], size=(3,))

            if (aug_index[0] == 1):
                x_image, y_mask = flipping(x_image, y_mask)

            if (aug_index[1] == 1):
                x_image, y_mask = rotating(x_image, y_mask)
            if (aug_index[2] == 1):
                x_image, y_mask = zooming(x_image, y_mask)
        # if (aug_index[3]==1):
        #     isseg = np.random.choice([True, False])
        #     x_image, y_mask = width_shift(x_image, y_mask, isseg)

            x_image = np.concatenate(x_image, axis=0).reshape(-1, 64, 64, 1)
            images.append(x_image)
            y_mask = np.concatenate(y_mask, axis=0).reshape(-1, 64, 64, 1)
            masks.append(y_mask)


        [images, masks]= [np.array(images),np.array(masks)]

        yield (images, masks)

def val_data_generator(x_train, y_train, batch_size):
    while True:
        images = []
        masks = []
        for i in range(batch_size):
            img_index = randrange(5)

            x_image = x_train[img_index]
            y_mask = y_train[img_index]
            aug_index = np.random.choice([0, 1], size=(3,))

            if (aug_index[0] == 1):
                x_image, y_mask = flipping(x_image, y_mask)

            if (aug_index[1] == 1):
                x_image, y_mask = rotating(x_image, y_mask)
            if (aug_index[2] == 1):
                x_image, y_mask = zooming(x_image, y_mask)
        # if (aug_index[3]==1):
        #     isseg = np.random.choice([True, False])
        #     x_image, y_mask = width_shift(x_image, y_mask, isseg)

            x_image = np.concatenate(x_image, axis=0).reshape(-1, 64, 64, 1)
            images.append(x_image)
            y_mask = np.concatenate(y_mask, axis=0).reshape(-1, 64, 64, 1)
            masks.append(y_mask)


        [images, masks]= [np.array(images),np.array(masks)]

        yield (images, masks)