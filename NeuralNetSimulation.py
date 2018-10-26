
# coding: utf-8


# encode pink noise 1D-time 1D-space with a linear conv neural net with one single type
# look at fft of filter after training

import numpy as np
import sys
theoption = int(sys.argv[1])

#This allows you to add to a list of results from another run.  To start from scratch use "results = []"
results = list(np.load('gridsearchresults.npy'))
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
from itertools import combinations_with_replacement
options = [1, 4, 9, 16, 36, 64, 144, 576]
#note that manual cell type allocations can be defined, see below line for an example
#Nneurons_options = [(36, 144, 36, 144), (37, 144, 35, 144), (38, 144, 34, 144), (39, 144, 33, 144), (36, 143, 36, 145), (36, 142, 36, 146), (36, 141, 36, 147), (35, 143, 37, 145), (34, 142, 38, 146), (33, 141, 39, 147)]
#default: try all combinations of cell types
Nneurons_options = list(combinations_with_replacement(options,NTYPE//2))
print(Nneurons_options)
print(len(Nneurons_options))

for NneuronsX in [Nneurons_options[theoption]]:
    #cell types are paired.  To avoid pairing, remove the line below, make the for loop
    #go over Nneurons instead of NneuronsX, and define Nneurons_options using 4-tuples instead
    #of 2-tuples
    Nneurons = NneuronsX + NneuronsX
    NTYPE = len(Nneurons)
    print('NTYPE', NTYPE)
    for log_noise in [-1.0]: #can experiment with multiple values
        for l1_val in [2500000*np.sqrt(10)]: #Can also experiment with multiple values.  NOTE: the actual l1 penalty scales as the reciprocal of this value in this implementation
            import urllib
            from urllib.request import urlretrieve
            from IPython.display import display
            from PIL import Image
            import numpy as np
            import tarfile
            from scipy.misc import imsave, comb
            from collections import Counter
            import matplotlib as mpl
            import h5py
            from skimage import io
            from matplotlib import gridspec
            import matplotlib.collections
            import matplotlib.patches as patches
            from collections import Counter
            import operator
            import imageio
            imageio.plugins.ffmpeg.download()
            import os
            import nibabel as nib
            from nibabel.testing import data_path
            import numpy as np
            import matplotlib.pyplot as plt
            from ipywidgets import interact, interactive, fixed, interact_manual
            import time
            from scipy import ndimage
            from scipy.io import loadmat, savemat
            import pickle
            import sys
            from sklearn.decomposition import PCA
            import pylab
            from mpl_toolkits.mplot3d import Axes3D
            from joblib import Parallel, delayed
            import multiprocessing
            import ipywidgets as widgets
            from moviepy.editor import VideoClip
            from moviepy.video.io.bindings import mplfig_to_npimage
            from moviepy.video.fx.all import crop
            from numpy import cross, eye, dot
            from sklearn.decomposition import NMF
            from scipy.cluster.hierarchy import dendrogram, linkage
            from scipy.cluster.hierarchy import fcluster
            from sklearn.manifold import TSNE
            from sklearn.manifold import MDS
            from scipy.stats.mstats import zscore
            import timeit
            import sklearn.cluster
            import scipy.signal
            import scipy.interpolate as interpolate


            import keras.backend as K
            K.set_image_data_format('channels_last')
            from keras.models import Sequential, Model
            from keras.layers import Conv2D, ZeroPadding2D, BatchNormalization, Input
            from keras.layers import Conv2DTranspose, Reshape, Activation, Cropping2D, Flatten
            from keras.layers.advanced_activations import LeakyReLU
            from keras.activations import relu
            from keras.initializers import RandomNormal
            from keras.optimizers import RMSprop, SGD, Adam
            import keras
            import tensorflow as tf

            import math


            # In[3]:

            class Deconv3D(keras.layers.Layer):
                def __init__(self, nb_filter, kernel_dims, output_shape, subsample, **kwargs):
                    self.nb_filter = nb_filter
                    self.kernel_dims = kernel_dims
                    self.strides = (1,) + subsample + (1,)
                    self.output_shape_ = output_shape
                    assert K.backend() == 'tensorflow'
                    super(Deconv3D, self).__init__(**kwargs)

                def build(self, input_shape):
                    assert len(input_shape) == 5
                    self.input_shape_ = input_shape
                    W_shape = self.kernel_dims + (self.nb_filter, input_shape[4], )
                    self.W = self.add_weight(shape=W_shape, initializer='glorot_uniform', name='{}_W'.format(self.name), trainable=True)
                    self.b = self.add_weight(shape=(1,1,1,self.nb_filter,), initializer='zero', name='{}_b'.format(self.name), trainable=True)
                    super(Deconv3D, self).build(input_shape)

                def compute_output_shape(self, input_shape):
                    return (input_shape[0], ) + self.output_shape_[1:]

                def call(self, x, mask=None):
                    return tf.nn.conv3d_transpose(x, self.W, output_shape=self.output_shape_, strides=self.strides, padding='SAME', name=self.name) + self.b



            N = 50000 # number of clips to generate

            ALPHA_X = 2
            ALPHA_T = 2

            NX = 24 # size in time
            NT = 10# size in space
            '''
            Pink noise generation
            X = np.zeros([50000, 32, 32, 20])

            for i in range(N):
                if i % 1000 == 0:
                    print(i // 1000)
                ## Create 3D ramp

                # space ramp (rotationaly invariant)
                fX = np.fft.fftfreq(NX,d=1)[:NX]
                fX[-1]=np.abs(fX[-1])
                gX,gY = np.meshgrid(fX,fX)
                descXY = np.sqrt(gX**2+gY**2)**(-ALPHA_X)
                descXY[0,:] = descXY[1, :]
                descXY[:,0] = descXY[:, 1]
                descXY[0, 0] = descXY[1, 1]

                # time ramp
                fT = np.fft.fftfreq(NT,d=1)[:NT//2+1]
                fT[-1]=np.abs(fT[-1])
                descT = np.abs((fT/10)**(-ALPHA_T)) / 100
                descT[0] = descT[1]

                # multiply space and time ramps together to get spatio-temporal profile
                descXYT = descXY.T[:,:,None]*descT.T


                ## GENERATE PINK NOISE CLIPS

                #generate white noise in time domain
                wn=np.random.randn(NX,NX,NT)
                #shaping in freq domain
                s = np.fft.rfftn(wn)
                #muliply spectrum with desired ramp
                fft_sim = s * descXYT
                # inverse fourier transform to get a movie
                vid = np.fft.irfftn(fft_sim)

                X[i, :, :, :] = vid

            X = np.array(X)
            #np.save('pinknoise3D.npy', X)
            #X = np.load('pinknoise3D.npy')
            #print(X.shape)

            X = np.reshape(X, [50000, 32, 32, 20, 1])
            X = X = X[:, :32, :32, 0:20, :]
            print(X.shape)
            np.save('pinknoise23D.npy', X)
            '''
            X = np.load('pinknoise23D.npy')


            split_frame = int(0.8*N)
            X = X[:, :24, :24, 0:10, :]


            X_train = X[:split_frame]
            y_train = X[:split_frame]

            X_test = X[split_frame:]
            y_test = X[split_frame:]

            print(X_test.shape,y_test.shape)



            from keras import backend as K
            from keras.engine.topology import Layer
            import numpy as np

            class MyBatchNorm(Layer):

                def __init__(self, **kwargs):
                    super(MyBatchNorm, self).__init__(**kwargs)

                def build(self, input_shape):
                    # Create a trainable weight variable for this layer.
                    super(MyBatchNorm, self).build(input_shape)  # Be sure to call this somewhere!

                def call(self, x):
                    x_L1_norm = Lambda(lambda x: K.sum(K.abs(x)))(x)
                    no_grad = K.stop_gradient(x_L1_norm)
                    returned = x #/ no_grad
                    return returned

                def compute_output_shape(self, input_shape):
                    return input_shape


            class L1(keras.regularizers.Regularizer):
                """Regularizer for L1 and L2 regularization.
                # Arguments
                    l1: Float; L1 regularization factor.
                    l2: Float; L2 regularization factor.
                """
                def __init__(self, l1):
                    self.l1 = K.cast_to_floatx(l1)
                def __call__(self, x):
                    regularization = 0.0
                    regularization += 1000000 * K.sum(K.mean(K.abs(x), axis=0)) / self.l1
                    return regularization
                def get_config(self):
                    return {'l1': float(self.l1)}


            class crazyReg(keras.regularizers.Regularizer):
                """Regularizer for L1 and L2 regularization.
                # Arguments
                    l1: Float; L1 regularization factor.
                    l2: Float; L2 regularization factor.
                """
                def __init__(self, l2):
                    self.l2 = K.cast_to_floatx(l2)
                def __call__(self, x):
                    regularization = 0
                    regularization += self.l2 *(K.sum(K.square(x)))
                    print('hey', x.shape)
                    return regularization
                def get_config(self):
                    return {'l1': float(self.l1)}


            KS = 20
            KT = 10
            NX = 24
            NY = NX
            NT = 10


            NNEURON = np.sum(Nneurons)


            from keras.layers import Input, Dense, Activation, LocallyConnected2D, ZeroPadding3D, Dropout, GaussianNoise, Lambda
            from keras.models import Model, Sequential
            from keras import regularizers
            from keras.layers.normalization import BatchNormalization
            from keras.layers.core import ActivityRegularization
            from keras.layers.convolutional import Conv2D, Conv3D
            from keras import constraints
            from keras import backend as K



            model = Sequential()



            model.add(Lambda(lambda x: K.spatial_3d_padding(x, ((19, 19), (19, 19), (7, 2))), input_shape = (NX,NY,NT,1)), )


            model.add(Conv3D(filters = NTYPE,
                             kernel_size = (KS,KS,KT),
                             strides=(1, 1, 1),
                             padding='valid',
                             data_format=None,
                             dilation_rate=(2, 2, 1),
                             activation=None,
                             use_bias=False,
                             kernel_initializer='glorot_uniform',
                             bias_initializer='zeros',
                             bias_regularizer=None,
                             kernel_regularizer=keras.regularizers.l2(50000),
                             activity_regularizer=None,
                             kernel_constraint=None,
                             bias_constraint=None))

            model.summary()
            
            model.add(Activation('relu'))

            model.add(Flatten())
            model.add(Dense(NX*NY*NT*NTYPE,use_bias=False,trainable=False,  activity_regularizer=L1(l1_val)))
            

            model.add(Reshape([NX, NY, NT, NTYPE]))



            model.add(GaussianNoise(10**log_noise))



            model.add(Lambda(lambda x: K.spatial_3d_padding(x, ((10, 9), (10, 9), (7, 2))), input_shape = (NX,NY,NT,1)), )

            model.add(Conv3D(filters = 1,
                             kernel_size = (KS,KS,KT),
                             input_shape = (NX,NY,NT,1),
                             strides=(1, 1, 1),
                             padding='valid',
                             data_format=None,
                             dilation_rate=(1,1,1),
                             activation=None,
                             use_bias=False,
                             kernel_initializer='glorot_uniform',
                             bias_initializer='zeros',
                             bias_regularizer=None,
                             kernel_regularizer=keras.regularizers.l1(0),
                             activity_regularizer=None,
                             kernel_constraint=None,
                             bias_constraint=None))

            model.summary()
            model.add(Reshape((NX,NY,NT,1)))

            model.summary()


            # REGULAR LATTICE of neurons

            W = np.zeros((NX*NY*NT*NTYPE,NX*NY*NT*NTYPE))

            pointer = 0
            for typ in range(NTYPE):
                density = np.sqrt(NX * NY / Nneurons[typ])
                ycount = typ
                xcount = typ
                totalcount = 0
                for y in range(NY):
                    if ycount % density - density < -1:
                        ycount += 1
                    elif ycount % density - density >= -1:
                        ycount += 1
                        xcount = 0
                        for x in range(NX):
                            if xcount % density >= 1:
                                xcount += 1
                            elif xcount % density < 1:
                                xcount += 1
                                totalcount +=1
                                for t in range(NT):
                                    W[typ + NTYPE*t + NTYPE*NT*(NX * y + x), typ + NTYPE*t + NTYPE*NT*(NX * y + x)] = 1
                print('Total', totalcount, ' Out of ', Nneurons[typ])
                
                #Randomly fill in neurons if the number of neurons doesn't exactly fit a regular lattice
                Px = np.random.permutation(NX*NY)
                for t in range(NT):
                    for x in range(Nneurons[typ] - totalcount):
                        W[typ + NTYPE*t + NTYPE*NT*Px[x], typ + NTYPE*t + NTYPE*NT*Px[x]] = 1 #
                        pointer = pointer + 1
            

            print(pointer)
            model.layers[4].set_weights((W,))



            def mse_center(y_true, y_pred):
                print(y_pred.shape)
                return K.mean(K.square(y_pred[:,:,:] - y_true[:,:,:]), axis=-1)


            model.compile(loss= mse_center,
                          optimizer='adam',
                         metrics = ['mean_squared_error'])
            print('hi')


            hist = model.fit(  X_train,
                        y_train,
                        batch_size=100,
                        validation_data=(X_test, y_test),
                        epochs=10)
            print('hi')


            W = np.squeeze(model.layers[1].get_weights()[0])
            W = np.expand_dims(W,axis = 3)





            layer_output = model.layers[4].output
            layer_out_func = K.function([model.input], [layer_output])
            layerout = layer_out_func([X_test[:2000]])[0]




            total_act_cell_type = []
            avg_act_cell_type = []
            total_act = 0
            
            num_neurons = 0
            for i in range(NTYPE):
                num_neurons += Nneurons[i]
            curr = 0
            for i in range(NTYPE):

                avg_act_cell_type.append(np.sum(np.mean(np.abs(layerout[:, i:-1:NTYPE]), axis=0)) / Nneurons[i])
            print('TOTAL_ACT', total_act)
            total_act = np.sum(np.mean(np.abs(layerout[:, :]), axis=0))
            results.append({'RELU':'ON', 'NTYPE': NTYPE, 'Nneurons': Nneurons, 'log_noise': log_noise, 'l1_val': l1_val, 'KS': KS, 'total_act': total_act, 'total_act_cell_type': total_act_cell_type, 'avg_act_cell_type': avg_act_cell_type, 'W': W, 'MSE': hist.history['mean_squared_error'], 'VMSE': hist.history['val_mean_squared_error'], 'loss': hist.history['loss'], 'val_loss': hist.history['val_loss']})

            np.save('gridsearchresults.npy', results)
