# In the name of God
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
import numpy as np
from keras import backend as K
#K.tensorflow_backend._get_available_gpus()
from devol_Evo3D_HEART import DEvol
from genome_handler_Evo3D_HEART import  GenomeHandler
from metrics import *
import time
from keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage.measure import find_contours
import keras.losses
keras.losses.dice_coef_loss=dice_coef_loss
import keras.metrics
keras.metrics.dice_coef=dice_coef
from keras.preprocessing.image import  array_to_img

#(x_train, y_train), (x_test, y_test) = mnist.load_data()
def make_plots(X, y, y_pred):
    # PLotting the results'
    n_best = 20
    n_worst = 20
    img_rows = X.shape[1]
    img_cols = img_rows
    axis = tuple(range(1, X.ndim))
    scores = numpy_dice(y, y_pred, axis=axis)
    print(scores)
    sort_ind = np.argsort(scores)[::-1]
    indice = np.nonzero(y.sum(axis=axis))[0]
    # Add some best and worst predictions
    img_list = []
    count = 1
    for ind in sort_ind:
        if ind in indice:
            img_list.append(ind)
            count += 1
        if count > n_best:
            break

    segm_pred = y_pred[img_list].reshape(-1, img_rows, img_cols)
    img = X[img_list].reshape(-1, img_rows, img_cols)
    segm = y[img_list].reshape(-1, img_rows, img_cols).astype('float32')
    n_cols = 4
    n_rows = int(np.ceil(len(img) / n_cols))

    fig = plt.figure(figsize=[4 * n_cols, int(4 * n_rows)])
    gs = gridspec.GridSpec(n_rows, n_cols)

    for mm in range(len(img)):

        ax = fig.add_subplot(gs[mm])
        ax.imshow(img[mm], 'gray')
        contours = find_contours(segm[mm], 0.05, fully_connected='high')
        for n, contour in enumerate(contours):
            ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color='m')

        contours = find_contours(segm_pred[mm], 0.05, fully_connected='high')
        for n, contour in enumerate(contours):
            ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color='y')

        ax.axis('image')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect(1)  # aspect ratio of 1

    fig.savefig('best_predictions.png', bbox_inches='tight', dpi=300)

    img_list = []
    count = 1
    for ind in sort_ind[::-1]:
        if ind in indice:
            img_list.append(ind)
            count += 1
        if count > n_worst:
            break

    segm_pred = y_pred[img_list].reshape(-1, img_rows, img_cols)
    img = X[img_list].reshape(-1, img_rows, img_cols)
    segm = y[img_list].reshape(-1, img_rows, img_cols).astype('float32')

    n_cols = 4
    n_rows = int(np.ceil(len(img) / n_cols))

    fig = plt.figure(figsize=[4 * n_cols, int(4 * n_rows)])
    gs = gridspec.GridSpec(n_rows, n_cols)

    for mm in range(len(img)):

        ax = fig.add_subplot(gs[mm])
        ax.imshow(img[mm], 'gray')
        contours = find_contours(segm[mm], 0.05, fully_connected='high')
        for n, contour in enumerate(contours):
            ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color='m')

        contours = find_contours(segm_pred[mm], 0.05, fully_connected='high')
        for n, contour in enumerate(contours):
            ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color='y')

        ax.axis('image')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect(1)  # aspect ratio of 1

    fig.savefig('worst_predictions.png', bbox_inches='tight', dpi=300)


# x_train=np.load("x_train.npy")
# y_train=np.load("y_train.npy")
# x_test=np.load("x_test.npy")
# y_test=np.load("y_test.npy")
# x_val=np.load("x_val.npy")
# y_val=np.load("y_val.npy")


x_train=np.load("X_train3d.npy")
y_train=np.load("y_train3d.npy")
x_test=np.load("X_test3d.npy")
y_test=np.load("y_test3d.npy")
x_val=np.load("X_val3d.npy")
y_val=np.load("y_val3d.npy")


print(np.shape(y_train))

K.set_image_data_format("channels_last")

x_train = x_train.reshape(x_train.shape[0],32, 64, 64, 1).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 32, 64, 64, 1).astype('float32') / 255
x_val = x_val.reshape(x_val.shape[0], 32,  64, 64, 1).astype('float32') / 255

 
#y_train = to_categorical(y_train)
#y_test = to_categorical(y_test)
#y_val = to_categorical(y_val)

dataset = ((x_train, y_train), (x_test, y_test), (x_val, y_val))
start = time.time()
genome_handler = GenomeHandler(max_block_num=7,
                               #max_dense_layers=2, # includes final dense layer
                               max_filters=128,
                               #max_dense_nodes=64,
                               input_shape=x_train.shape[1:]
                               #n_classes=3)
                               )

devol = DEvol(genome_handler)
model = devol.run(dataset=dataset,
                  num_generations=1,
                  pop_size=50,
                  epochs=5,
                  )
model.summary()
end = time.time()

print('Elapsed time:', round((end - start) / 60, 2))





