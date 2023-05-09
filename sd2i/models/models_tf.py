import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import  Input, UpSampling2D, Reshape, Dense, Conv2D, Flatten, Cropping2D
from numpy import ceil

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import  Input, UpSampling2D, Reshape, Dense, Conv2D, Flatten, Cropping2D
from numpy import ceil

def SD2I(npix, factor=8, upsample = True):

    '''
    SD2I image reconstruction network with upsampling
    
    Inputs:
        npix: number of pixels in the image per each dimension
    
    '''
    
    xi = Input(shape=(1,))
    x = Flatten()(xi)
    
    
    if upsample:
        x = Dense(64, kernel_initializer='random_normal', activation='relu')(x)
        x = Dense(64, kernel_initializer='random_normal', activation='relu')(x)
        x = Dense(64, kernel_initializer='random_normal', activation='relu')(x)
        x = Dense(int(ceil(npix / 4)) * int(ceil(npix / 4)) * factor, kernel_initializer='random_normal', activation='linear')(x)

        x = Reshape((int(ceil(npix / 4)), int(ceil(npix / 4)), factor))(x)   
        
        x = UpSampling2D(size = (2,2))(x)

        if npix % 4 == 1 or npix % 4 == 2:
          x = Cropping2D(cropping=((0, 1), (0, 1)))(x)
          
        x = Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')(x)
        x = Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')(x)
        x = Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')(x)
        
        x = UpSampling2D(size = (2,2))(x)

        if npix % 2 == 1:
          x = Cropping2D(cropping=((1, 0), (1, 0)))(x)

        x = Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')(x)
        x = Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')(x)
        x = Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')(x)

        x = Conv2D(filters = 1, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'linear')(x)
    
    else:
        x = Dense(128, kernel_initializer='random_normal', activation='relu')(x)
        x = Dense(128, kernel_initializer='random_normal', activation='relu')(x)
        x = Dense(128, kernel_initializer='random_normal', activation='relu')(x)
        x = Dense(npix * npix * factor, kernel_initializer='random_normal', activation='linear')(x)
        
        x = Reshape((npix, npix, factor))(x)
        
        x = Conv2D(filters = 128, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')(x)

        x = Conv2D(filters = 128, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')(x)
        x = Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')(x)
        x = Conv2D(filters = 1, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'linear')(x)

    model = Model(xi, x)

    return(model)

def GANrec(npix):

    '''
    GANrec image reconstruction network
    
    Inputs:
        npix: number of pixels in the image per each dimension
    
    '''
    
    xi = Input(shape=(npix,npix,1))
    x = Flatten()(xi)
    
    x = Dense(256, kernel_initializer='random_normal', activation='relu')(x)
    x = Dense(256, kernel_initializer='random_normal', activation='relu')(x)
    x = Dense(256, kernel_initializer='random_normal', activation='relu')(x)
    x = Dense(npix*npix, kernel_initializer='random_normal', activation='relu')(x)

    
    x = Reshape((int(npix // 1), int(npix // 1), 1))(x)   
    
    x = Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')(x)
    x = Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')(x)
    x = Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')(x)
    x = Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')(x)
    x = Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')(x)
    x = Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')(x)
    x = Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')(x)
    x = Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')(x)
    x = Conv2D(filters = 1, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'linear')(x)
    
    model = Model(xi, x)

    return(model)

def Automap(npix, npr):

    '''
    Automap image reconstruction network
    
    Inputs:
        npix: number of pixels in the reconstructed image per each dimension (number of detector elements in sinograms)
        npr: number of tomographic angles (projections)
    
    '''
    

    xi = Input(shape=(npr,npix))
    x = Flatten()(xi)
    
    x = Dense(2*npix*npix, kernel_initializer='random_normal', activation='relu')(x)
    x = Dense(npix*npix, kernel_initializer='random_normal', activation='relu')(x)
    x = Dense(npix*npix, kernel_initializer='random_normal', activation='relu')(x)
    
    x = Reshape((int(npix // 1), int(npix // 1), 1))(x)   
    
    x = Conv2D(filters = 128, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')(x)
    x = Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')(x)
    x = Conv2D(filters = 1, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'linear')(x)
    
    
    model = Model(xi, x)

    return(model)

class Discriminator(tf.keras.Model):
    
    def __init__(self, npix, npr):
        super(Discriminator, self).__init__()

        self.conv_1 = Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')
        self.conv_2 = Conv2D(filters = 128, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')
        self.conv_3 = Conv2D(filters = 256, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')
        self.conv_4 = Conv2D(filters = 512, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')
        self.reshape = Reshape((npix * npr * 512, ))

    def call(self, inputs):
        tf.keras.backend.set_floatx('float32')
        x = self.conv_1(inputs)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        return self.reshape(x)
