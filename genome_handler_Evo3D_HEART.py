import numpy as np
import random as rand
import math
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten, Input, concatenate, Conv3D, Conv3DTranspose
from keras.layers.convolutional import Convolution2D, MaxPooling2D, SeparableConvolution2D,Conv2DTranspose, MaxPooling3D, AveragePooling3D, UpSampling3D
from keras.layers.normalization import BatchNormalization
from metrics import *
import keras.losses
keras.losses.dice_coef_loss=dice_coef_loss
import keras.metrics
keras.metrics.dice_coef=dice_coef
from keras.preprocessing.image import ImageDataGenerator
from keras.initializers import RandomNormal, VarianceScaling
from keras.models import Model
from keras.optimizers import adam,rmsprop,adadelta,adagrad
from keras.utils import multi_gpu_model
from DepthwiseConv3D import DepthwiseConv3D
import tensorflow as tf
from se import squeeze_excite_block



class ModelMGPU(Model):
    def __init__(self, model, gpus):
        pmodel = multi_gpu_model(model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = model

    def __getattribute__(self, attrname):
        '''Override load and save methods to be used from the serial-model. The
        serial-model holds references to the weights in the multi-gpu model.
        '''
        # return Model.__getattribute__(self, attrname)
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)

        return super(ModelMGPU, self).__getattribute__(attrname)
        


class GenomeHandler:

    """
    Defines the configuration and handles the conversion and mutation of
    individual genomes. Should be created and passed to a `DEvol` instance.

    ---
    Genomes are represented as fixed-with lists of integers corresponding
    to sequential layers and properties. A model with 2 convolutional layers
    and 1 dense layer would look like:

    [<conv layer><conv layer><dense layer><optimizer>]

    The makeup of the convolutional layers and dense layers is defined in the
    GenomeHandler below under self.convolutional_layer_shape and
    self.dense_layer_shape. <optimizer> consists of just one property.
    """

    def __init__(self, max_block_num,  max_filters, input_shape,
                 batch_normalization=True, dropout=True, max_pooling=True,
                 optimizers=None, activations=None, learningrates=None, batch_size=None, augsize=None, kernels=None):
        """
        Creates a GenomeHandler according 

        Args:
            max_conv_layers: The maximum number of convolutional layers           
            max_conv_layers: The maximum number of dense (fully connected)
                    layers, including output layer
            max_filters: The maximum number of conv filters (feature maps) in a
                    convolutional layer
            max_dense_nodes: The maximum number of nodes in a dense layer
            input_shape: The shape of the input
            n_classes: The number of classes
            batch_normalization (bool): whether the GP should include batch norm
            dropout (bool): whether the GP should include dropout
            max_pooling (bool): whether the GP should include max pooling layers
            optimizers (list): list of optimizers to be tried by the GP. By
                    default, the network uses Keras's built-in adam, rmsprop,
                    adagrad, and adadelta
            activations (list): list of activation functions to be tried by the
                    GP. By default, relu and sigmoid.
        """
        # if max_dense_layers < 1:
        #     raise ValueError(
        #         "At least one dense layer is required for softmax layer"
        #     )
        if max_filters > 0:
            filter_range_max = int(math.log(max_filters, 2)) + 1
        else:
            filter_range_max = 0
        self.optimizer = optimizers or [
            'adam',
            'rmsprop',
            'adagrad',
            'adadelta'
        ]
        self.kernel = kernels or [
            'RandomNormal',
            'RandomUniform',
            'TruncatedNormal',
            'VarianceScaling',
            'glorot_normal',
            'glorot_uniform',
            'he_normal',
            'he_uniform'
            
        ] 
        self.learningrate = learningrates or [
            0.1,
            0.01,
            0.001,
            0.0001
        ]
        self.batchsize = batch_size or [
            4,
            8,
            16,
            

        ]
        self.augmentationsize = augsize or [
            4000,
            8000,
            16000,
            32000,
           

        ]
        self.activation = activations or [
            'relu',
            'sigmoid',
            'tanh',
            'elu'
        ]
        self.convolutional_layer_shape = [
            "active",
            "shortcon",
            # "typeshortcon",
            "longcon",
            # "typelongcon",
            "convtype",
            "conv1",
            "conv size1",
            "conv2",
            "conv size2",
            "conv3",
            "conv size3",
            "conv4",
            "conv size4",
            "conv5",
            "conv size5",

            # "conv",
            "num filters",
            "batch normalization",
            "activation",
            "dropout",
            "pooling",
        ]
        # self.dense_layer_shape = [
        #     "active",
        #     "num nodes",
        #     "batch normalization",
        #     "activation",
        #     "dropout",
        # ]
        self.layer_params = {

            "active": [0, 1],
            "shortcon": [0,1],
            #"typeshortcon": [0,1], # 0=elementwise sum, 1=concatenation
            "longcon":list(range(0, 6)),
            "convtype": [0,1],
            #"typelongcon":[0,1], # 0=elementwise sum, l=concatenation
 
            # "typelongcon":[0,1], # 0=elementwise sum, l=concatenation
            "conv1": [0, 1],
            # "conv": list(range(len(self.conv))),
            "conv size1": [1, 3, 5, 7],  # 3= 3*3 filter size, 5=5*5 filter size
            "conv2": [0, 1],
            # "conv": list(range(len(self.conv))),
            "conv size2": [1, 3, 5, 7],  # 3= 3*3 filter size, 5=5*5 filter size
            "conv3": [0, 1],
            # "conv": list(range(len(self.conv))),
            "conv size3": [1, 3, 5, 7],  # 3= 3*3 filter size, 5=5*5 filter size
            "conv4": [0, 1],
            # "conv": list(range(len(self.conv))),
            "conv size4": [1, 3, 5, 7],  # 3= 3*3 filter size, 5=5*5 filter size
            "conv5": [0, 1],
            # "conv": list(range(len(self.conv))),
            "conv size5": [1, 3, 5, 7],  # 3= 3*3 filter size, 5=5*5 filter size
            "num filters": [2**i for i in range(3, filter_range_max)],
            # "num nodes": [2**i for i in range(4, int(math.log(max_dense_nodes, 2)) + 1)],
            "batch normalization": [0, (1 if batch_normalization else 0)],
            "activation": list(range(len(self.activation))),
            "dropout": [(i if dropout else 0) for i in range(7)],
            #"max pooling": list(range(3)) if max_pooling else 0,
            "pooling":[0,1], # 1=maxpooling, 0=averagepooling
        }
        

        self.convolution_layers = max_block_num
        self.convolution_layer_size = len(self.convolutional_layer_shape)
        # self.dense_layers = max_dense_layers - 1 # this doesn't include the softmax layer, so -1
        # self.dense_layer_size = len(self.dense_layer_shape)
        self.input_shape = input_shape
        # self.n_classes = n_classes
        

    def convParam(self, i):
        key = self.convolutional_layer_shape[i]
        return self.layer_params[key]
        

    # def denseParam(self, i):
    #     key = self.dense_layer_shape[i]
    #     return self.layer_params[key]
    #     print("third")

    def mutate(self, genome, num_mutations):
        num_mutations = np.random.choice(num_mutations)
        for i in range(num_mutations):
            index = np.random.choice(list(range(1, len(genome))))
            if index < self.convolution_layer_size * self.convolution_layers:
                if genome[index - index % self.convolution_layer_size]:
                    range_index = index % self.convolution_layer_size
                    choice_range = self.convParam(range_index)
                    genome[index] = np.random.choice(choice_range)
                elif rand.uniform(0, 1) <= 0.01: # randomly flip deactivated layers
                    genome[index - index % self.convolution_layer_size] = 1
            elif index == 114:
                genome[index] = np.random.choice(list(range(len(self.optimizer))))

            elif index == 115:
                genome[index] = np.random.choice(list(range(len(self.learningrate))))

            elif index == 116:
                genome[index] = np.random.choice(list(range(len(self.batchsize))))
            elif index == 117:
                genome[index] = np.random.choice(list(range(len(self.augmentationsize))))
            elif index == 118:
                genome[index] = np.random.choice(list(range(len(self.kernel))))
        
        return genome

    def conv_block (self, m, ct, dim, acti, bn, att, lc, dp, cn1, cs1, cn2, cs2, cn3, cs3, cn4, cs4, cn5, cs5, active, k):

        if active and (cn1 or cn2 or cn3 or cn4 or cn5):
            if k== 0:
                 init = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
            elif k == 1:
                init = keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)
            elif k == 2:
                init = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)
            elif k== 3:
                init = VarianceScaling(scale=1.0 / 9.0)
            elif k == 4:
                init = keras.initializers.glorot_normal(seed=None)
            elif k == 5:
                init = keras.initializers.glorot_uniform(seed=None)
            elif k == 6:
                init = keras.initializers.he_normal(seed=None)
            elif k == 7:
                init = keras.initializers.he_uniform(seed=None)
                
            
            init = VarianceScaling(scale=1.0 / 9.0)
            n = m
            n1 = K.zeros(tf.shape(m), dtype=m.dtype)
            n2 = K.zeros(tf.shape(m), dtype=m.dtype)
            n3 = K.zeros(tf.shape(m), dtype=m.dtype)
            n4 = K.zeros(tf.shape(m), dtype=m.dtype)
            n5 = K.zeros(tf.shape(m), dtype=m.dtype)
            # n1 = K.eval(n1)
            if ct ==0:
                if cn1 == 1:
                    n = Conv3D(dim, (cs1, cs1, cs1), activation=acti, padding='same', kernel_initializer=init)(m)
                    n = BatchNormalization()(n) if bn else n
                    n1 = n
                if cn2 == 1:
                    n = Conv3D(dim, (cs2, cs2, cs2), activation=acti, padding='same', kernel_initializer=init)(n)
                    n = BatchNormalization()(n) if bn else n
                    n2 = n
                if cn3 == 1:
                    n = Conv3D(dim, (cs3, cs3, cs3), activation=acti, padding='same', kernel_initializer=init)(n)
                    n = BatchNormalization()(n) if bn else n
                    n3 = n
                if cn4 == 1:
                    n = Conv3D(dim, (cs4, cs4, cs4), activation=acti, padding='same', kernel_initializer=init)(n)
                    n = BatchNormalization()(n) if bn else n
                    n4 = n
                if cn5 == 1:
                    n = Conv3D(dim, (cs5, cs5, cs5), activation=acti, padding='same', kernel_initializer=init)(n)
                    n = BatchNormalization()(n) if bn else n
                    n5 = n
            elif ct==1:            
                
                if cn1 == 1:
                    n = Conv3D(dim, (cs1, cs1, cs1), activation=acti, dilation_rate=(2, 2, 2), padding='same', kernel_initializer=init)(m)
                    n = BatchNormalization()(n) if bn else n
                    n1 = n
                if cn2 == 1:
                    n = Conv3D(dim, (cs2, cs2, cs2), activation=acti, dilation_rate=(2, 2, 2),padding='same', kernel_initializer=init)(n)
                    n = BatchNormalization()(n) if bn else n
                    n2 = n
                if cn3 == 1:
                    n = Conv3D(dim, (cs3, cs3, cs3), activation=acti, dilation_rate=(2, 2, 2),padding='same', kernel_initializer=init)(n)
                    n = BatchNormalization()(n) if bn else n
                    n3 = n
                if cn4 == 1:
                    n = Conv3D(dim, (cs4, cs4, cs4), activation=acti, dilation_rate=(2, 2, 2),padding='same', kernel_initializer=init)(n)
                    n = BatchNormalization()(n) if bn else n
                    n4 = n
                if cn5 == 1:
                    n = Conv3D(dim, (cs5, cs5, cs5), activation=acti, dilation_rate=(2, 2, 2),padding='same', kernel_initializer=init)(n)
                    n = BatchNormalization()(n) if bn else n
                    n5 = n

            
            
            n = Dropout(float(dp / 10.0))(n)
            if att:
                return concatenate([n,m], axis=4)
        # elif res and tsc==0:
        #    return n #keras.layers.Add()([n,m])
            else:
                return n
        else:
            return m

    

    def level_block(self, m, genome, depth, up, offset):


        if depth > 1:


            active = genome[offset]
            att = genome[offset + 1]  # short Connection
            # tsc = genome[offset + 2]  # type of shortcut connection, elementwise sum or concatenation
            lc = genome[offset + 2]  # long connection
            ct = genome[offset + 3]
            # tlc = genome[offset + 4]  # type of long conecction, elementwise sum or concatenation
            cn1 = genome[offset + 4]  # first conv layer
            cs1 = genome[offset + 5]  # first conv layer filter size
            cn2 = genome[offset + 6]  # second conv layer
            cs2 = genome[offset + 7]  # second conv layer filter size
            cn3 = genome[offset + 8]  # third conv layer
            cs3 = genome[offset + 9]  # third conv layer lize
            cn4 = genome[offset + 10]  # fourth conv layer
            cs4 = genome[offset + 11]  # fourth conv layer filter size
            cn5 = genome[offset + 12]  # fifth conv layer
            cs5 = genome[offset + 13]  # fifth conv layer filter size
            dim = genome[offset + 14]  # the number of filters
            bn = genome[offset + 15]  # the Batch normalization
            ac = genome[offset + 16]  # Activation functions
            dp = genome[offset + 17]  # the dropout
            pl = genome[offset + 18]  # type of pooling, maxpooling=1 or average pooling=0
            if ac == 0:
                acti = 'relu'
            elif ac ==1 :
                acti = 'sigmoid'
            elif ac ==2 :
                acti = 'tanh'
            else:
                acti = 'elu'
            k = genome[118]
            n= self.conv_block(m, ct, dim, acti, bn, att, lc, dp, cn1, cs1, cn2, cs2, cn3, cs3, cn4, cs4, cn5, cs5, active, k)
            if pl==1 and active and (cn1 or cn2 or cn3 or cn4 or cn5):
                m = MaxPooling3D()(n)
            elif pl==0 and active and (cn1 or cn2 or cn3 or cn4 or cn5):
                m = AveragePooling3D()(n)
            offset += self.convolution_layer_size
            # offset1=offset

            m = self.level_block( m, genome, depth-1, up, offset)

            if up :

                m = UpSampling3D()(m)
                m = Conv3D(dim, 2, activation=acti, padding='same')(m)
            else:


                offset -= self.convolution_layer_size
                active = genome[offset]
                if active and (cn1 or cn2 or cn3 or cn4 or cn5):
                    att = genome[offset + 1]  # short Connection
                    # tsc = genome[offset + 2]  # type of shortcut connection, elementwise sum or concatenation
                    lc = genome[offset + 2]  # long connection
                    ct = genome[offset + 3]
                    # tlc = genome[offset + 4]  # type of long conecction, elementwise sum or concatenation
                    cn1 = genome[offset + 4]  # first conv layer
                    cs1 = genome[offset + 5]  # first conv layer filter size
                    cn2 = genome[offset + 6]  # second conv layer
                    cs2 = genome[offset + 7]  # second conv layer filter size
                    cn3 = genome[offset + 8]  # third conv layer
                    cs3 = genome[offset + 9]  # third conv layer lize
                    cn4 = genome[offset + 10]  # fourth conv layer
                    cs4 = genome[offset + 11]  # fourth conv layer filter size
                    cn5 = genome[offset + 12]  # fifth conv layer
                    cs5 = genome[offset + 13]  # fifth conv layer filter size
                    dim = genome[offset + 14]  # the number of filters
                    bn = genome[offset + 15]  # the Batch normalization
                    ac = genome[offset + 16]  # Activation functions
                    dp = genome[offset + 17]  # the dropout
                    pl = genome[offset + 18]  # type of pooling, maxpooling=1 or average pooling=0
                    if ac == 0:
                        acti = 'relu'
                    elif ac ==1 :
                        acti = 'sigmoid'
                    elif ac ==2 :
                        acti = 'tanh'
                    else:
                        acti = 'elu'

                #dim= int(m.shape[-1])
                    m = Conv3DTranspose(dim, (3,3,3), strides=(2,2,2), activation=acti, padding='same')(m)


            #offset -= self.convolution_layer_size
            active=genome[offset]
            if active and (cn1 or cn2 or cn3 or cn4 or cn5):
                att = genome[offset + 1]  # short Connection
                # tsc = genome[offset + 2]  # type of shortcut connection, elementwise sum or concatenation
                lc = genome[offset + 2]  # long connection
                ct = genome[offset + 3]
                # tlc = genome[offset + 4]  # type of long conecction, elementwise sum or concatenation
                cn1 = genome[offset + 4]  # first conv layer
                cs1 = genome[offset + 5]  # first conv layer filter size
                cn2 = genome[offset + 6]  # second conv layer
                cs2 = genome[offset + 7]  # second conv layer filter size
                cn3 = genome[offset + 8]  # third conv layer
                cs3 = genome[offset + 9]  # third conv layer lize
                cn4 = genome[offset + 10]  # fourth conv layer
                cs4 = genome[offset + 11]  # fourth conv layer filter size
                cn5 = genome[offset + 12]  # fifth conv layer
                cs5 = genome[offset + 13]  # fifth conv layer filter size
                dim = genome[offset + 14]  # the number of filters
                bn = genome[offset + 15]  # the Batch normalization
                ac = genome[offset + 16]  # Activation functions
                dp = genome[offset + 17]  # the dropout
                pl = genome[offset + 18]  # type of pooling, maxpooling=1 or average pooling=0
                if ac == 0:
                    acti = 'relu'
                elif ac ==1 :
                    acti = 'sigmoid'
                elif ac ==2 :
                    acti = 'tanh'
                else:
                    acti = 'elu'

                k = genome[118]

                if lc ==0 : # without long connection
                    n = m
                elif lc==1:  # simple concatenation
                    n = concatenate([n, m], axis=4) #keras.layers.Add()([n,m])
                elif lc ==2:   # Dense block
                    init = VarianceScaling(scale=1.0 / 9.0)
                    n = m
                    n1 = K.zeros(tf.shape(m), dtype=m.dtype)
                    n2 = K.zeros(tf.shape(m), dtype=m.dtype)
                    n3 = K.zeros(tf.shape(m), dtype=m.dtype)
                    # n1 = K.eval(n1)

                    n = Conv3D(64, (1, 1,1), activation=acti, padding='same', dilation_rate=(2, 2, 2), kernel_initializer=init)(m)
                    n = BatchNormalization()(n) if bn else n
                    n1 = n
                    n1 = concatenate([n1, m], axis=4)
                    n = Conv3D(64, (3, 3, 3), activation=acti, padding='same', dilation_rate=(2, 2, 2), kernel_initializer=init)(n1)
                    n = BatchNormalization()(n) if bn else n
                    n2 = n
                    n2 = concatenate([n2, n1], axis=4)

                    n = Conv3D(64, (5, 5, 5), activation=acti, padding='same', dilation_rate=(2, 2, 2), kernel_initializer=init)(n2)
                    n = BatchNormalization()(n) if bn else n
                    n3 = n
                    n3 = concatenate([n3, n2], axis=4)

                    n = Dropout(float(dp / 10.0))(n3)

                    # elif res and tsc==0:
                    #    return n #keras.layers.Add()([n,m])

                elif lc ==3 :
                    init = VarianceScaling(scale=1.0 / 9.0)
                    n = Conv3D(64, (1, 1, 1), activation=acti, padding='same', dilation_rate=(2, 2, 2), kernel_initializer=init)(m)
                    n = BatchNormalization()(n) if bn else n

                    n = Conv3D(64, (3, 3, 3), activation=acti, padding='same', dilation_rate=(2, 2, 2), kernel_initializer=init)(n)
                    n = BatchNormalization()(n) if bn else n

                    n = Conv3D(64, (5, 5, 5), activation=acti, padding='same', dilation_rate=(2, 2, 2), kernel_initializer=init)(n)
                    n = BatchNormalization()(n) if bn else n

                    dim_n = np.shape(n)  # output
                    dim_m = np.shape(m)  # input
                    dim1 = dim_n[4]
                    dim2 = dim_m[4]

                    if dim_n[4] > dim_m[4]:
                        m = Conv3D(int(dim1), (1, 1, 1), activation=acti, padding='same', kernel_initializer=init)(m)
                        m = BatchNormalization()(m) if bn else m

                    elif dim_m[4] > dim_n[4]:
                        n = Conv3D(int(dim2), (1, 1, 1), activation=acti, padding='same', kernel_initializer=init)(n)
                        n = BatchNormalization()(n) if bn else n
                    # print(np.shape(n))
                    # print(np.shape(m))
                    n = keras.layers.Add()([n, m])
                    n = Dropout(float(dp / 10.0))(n)

                elif lc == 4:
                    n = squeeze_excite_block(m)

                elif lc == 5:
                    init = VarianceScaling(scale=1.0 / 9.0)
                    n = Conv3D(64, (3, 3, 3), activation=acti, padding='same', dilation_rate=(2, 2, 2), kernel_initializer=init)(m)
                    n = BatchNormalization()(n) if bn else n

                    n = Conv3D(64, (5, 5, 5), activation=acti, padding='same', dilation_rate=(2, 2, 2), kernel_initializer=init)(n)
                    n = BatchNormalization()(n) if bn else n

                    n = Conv3D(64, (7, 7, 7), activation=acti, padding='same', dilation_rate=(2, 2, 2), kernel_initializer=init)(n)
                    n = BatchNormalization()(n) if bn else n
                    n = concatenate([m, n], axis=4)

                    m = Conv3D(64, (1, 1, 1), activation=acti, padding='same', dilation_rate=(2, 2, 2),  kernel_initializer=init)(m)
                    dim_n = np.shape(n)  # output
                    dim_m = np.shape(m)  # input
                    dim1 = dim_n[4]
                    dim2 = dim_m[4]
                    if dim_n[4] > dim_m[4]:
                        m = Conv3D(int(dim1), (1, 1, 1), activation=acti, padding='same', kernel_initializer=init)(m)
                        m = BatchNormalization()(m) if bn else m

                    elif dim_m[4] > dim_n[4]:
                        n = Conv3D(int(dim2), (1, 1, 1), activation=acti, padding='same', kernel_initializer=init)(n)
                        n = BatchNormalization()(n) if bn else n
                    # print(np.shape(n))
                    # print(np.shape(m))
                    n = keras.layers.Add()([n, m])
                    n = Dropout(float(dp / 10.0))(n)


                m = self.conv_block(n, ct, dim, acti, bn, att, lc, dp, cn1, cs1, cn2, cs2, cn3, cs3, cn4, cs4, cn5, cs5, active, k)
                

        else:

            active=genome[offset]
            if active:
                att = genome[offset + 1]  # short Connection
                # tsc = genome[offset + 2]  # type of shortcut connection, elementwise sum or concatenation
                lc = genome[offset + 2]  # long connection
                ct = genome[offset + 3]
                # tlc = genome[offset + 4]  # type of long conecction, elementwise sum or concatenation
                cn1 = genome[offset + 4]  # first conv layer
                cs1 = genome[offset + 5]  # first conv layer filter size
                cn2 = genome[offset + 6]  # second conv layer
                cs2 = genome[offset + 7]  # second conv layer filter size
                cn3 = genome[offset + 8]  # third conv layer
                cs3 = genome[offset + 9]  # third conv layer lize
                cn4 = genome[offset + 10]  # fourth conv layer
                cs4 = genome[offset + 11]  # fourth conv layer filter size
                cn5 = genome[offset + 12]  # fifth conv layer
                cs5 = genome[offset + 13]  # fifth conv layer filter size
                dim = genome[offset + 14]  # the number of filters
                bn = genome[offset + 15]  # the Batch normalization
                ac = genome[offset + 16]  # Activation functions
                dp = genome[offset + 17]  # the dropout
                pl = genome[offset + 18]  # type of pooling, maxpooling=1 or average pooling=0
                if ac == 0:
                    acti = 'relu'
                elif ac ==1 :
                    acti = 'sigmoid'
                elif ac ==2 :
                    acti = 'tanh'
                else:
                    acti = 'elu'
                k = genome[118]
                m = self.conv_block(m, ct, dim, acti, bn, att, lc, dp, cn1, cs1, cn2, cs2, cn3, cs3, cn4, cs4, cn5, cs5, active, k)
                offset -= self.convolution_layer_size
            #m = self.conv_block(m, 32, 'relu', True, True, 10, 3, 3, 1)

        return m
    
    def EvoUNet(self, genome,  depth, upconv= False):
        out_ch = 1
        img_shape=(32,64,64,1)
        i = Input(shape=img_shape)
        o= self.level_block(i, genome, depth, upconv,0)
        o = Conv3D(out_ch, (1,1,1), activation='sigmoid')(o)
        return Model(inputs=i, outputs=o)

    def decode(self, genome):
        # if not self.is_compatible_genome(genome):
        #     raise ValueError("Invalid genome for specified configs")

        print(genome)
        model = self.EvoUNet(genome,  7, upconv=False)

        model.summary()
        pl_model = model
        
        op = self.optimizer[genome[114]]
        batch = self.batchsize[genome[116]]
        aug = self.augmentationsize[genome[117]]
        print(op)
        if op=='adam':
            
            
            pl_model.compile(optimizer=adam(lr=self.learningrate[genome[115]]), loss= dice_coef_loss,metrics=[dice_coef])
            
        elif op=='rmsprop':
            
                          
            pl_model.compile(optimizer=rmsprop(lr=self.learningrate[genome[115]]), loss=dice_coef_loss, metrics=[dice_coef])
        elif op=='adadelta':

            
            pl_model.compile(optimizer=adadelta(lr=self.learningrate[genome[115]]), loss=dice_coef_loss, metrics=[dice_coef])
        else:
            
            pl_model.compile(optimizer=adagrad(lr=self.learningrate[genome[115]]), loss=dice_coef_loss, metrics=[dice_coef])
            
        
        
        return pl_model, model, batch, aug

    def genome_representation(self):
        encoding = []
        for i in range(self.convolution_layers):
            for key in self.convolutional_layer_shape:
                encoding.append("Conv" + str(i) + " " + key)
        # for i in range(self.dense_layers):
        #     for key in self.dense_layer_shape:
        #         encoding.append("Dense" + str(i) + " " + key)
        encoding.append("Optimizer")
        encoding.append("Learning Rate")
        encoding.append("Batch Size")
        encoding.append("Augmentation Size")
        encoding.append("Initializer")

        return encoding

    def generate(self):
        genome = []
        for i in range(self.convolution_layers):
            for key in self.convolutional_layer_shape:
                param = self.layer_params[key]
                genome.append(np.random.choice(param))
        # for i in range(self.dense_layers):
        #     for key in self.dense_layer_shape:
        #         param = self.layer_params[key]
        #         genome.append(np.random.choice(param))
        genome.append(np.random.choice(list(range(len(self.optimizer)))))
        genome.append(np.random.choice(list(range(len(self.learningrate)))))
        genome.append(np.random.choice(list(range(len(self.batchsize)))))
        genome.append(np.random.choice(list(range(len(self.augmentationsize)))))
        genome.append(np.random.choice(list(range(len(self.kernel)))))
        genome[0] = 1
        #genome[40] = 1
        

        return genome


    def best_genome(self, csv_path, metric='accuracy', include_metrics=True):
        best = max if metric is 'accuracy' else min
        col = -1 if metric is 'accuracy' else -2
        data = np.genfromtxt(csv_path, delimiter=",")
        row = list(data[:, col]).index(best(data[:, col]))
        genome = list(map(int, data[row, :-2]))
        if include_metrics:
            genome += list(data[row, -2:])
        
        return genome

    def decode_best(self, csv_path, metric='accuracy'):
        
        return self.decode(self.best_genome(csv_path, metric, False))
