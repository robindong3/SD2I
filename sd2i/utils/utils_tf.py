

from numpy import less, greater, Inf
from numpy.random import rand, shuffle

import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import tensorflow_addons as tfa
import math as m
import numpy as np

def tf_gpu_devices():
        
    if tf.test.gpu_device_name(): 
        print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))
    else:
       print("Please install GPU version of TF")

class ReduceLROnPlateau_custom(Callback):

    '''
    Custom reduce learning rate on plateau callback, it can be used in custom training loops
    '''
    
    def __init__(self,
                  ## Custom modification:  Deprecated due to focusing on validation loss
                  # monitor='val_loss',
                  factor=0.5,
                  patience=10,
                  verbose=0,
                  mode='auto',
                  min_delta=1e-4,
                  cooldown=0,
                  min_lr=0,
                  sign_number = 4,
                  ## Custom modification: Passing optimizer as arguement
                  optim_lr = None,
                  ## Custom modification:  linearly reduction learning
                  reduce_lin = False,
                  **kwargs):
    
        ## Custom modification:  Deprecated
        # super(ReduceLROnPlateau, self).__init__()
    
        ## Custom modification:  Deprecated
        # self.monitor = monitor
        
        ## Custom modification: Optimizer Error Handling
        if tf.is_tensor(optim_lr) == False:
            raise ValueError('Need optimizer !')
        if factor >= 1.0:
            raise ValueError('ReduceLROnPlateau ' 'does not support a factor >= 1.0.')
        ## Custom modification: Passing optimizer as arguement
        self.optim_lr = optim_lr  
    
        self.factor = factor
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.wait = 0
        self.best = 0
        self.mode = mode
        self.monitor_op = None
        self.sign_number = sign_number
        
    
        ## Custom modification: linearly reducing learning
        self.reduce_lin = reduce_lin
        self.reduce_lr = True
        
    
        self._reset()

    def _reset(self):
         """Resets wait counter and cooldown counter.
         """
         if self.mode not in ['auto', 'min', 'max']:
             print('Learning Rate Plateau Reducing mode %s is unknown, '
                             'fallback to auto mode.', self.mode)
             self.mode = 'auto'
         if (self.mode == 'min' or
                 ## Custom modification: Deprecated due to focusing on validation loss
                 # (self.mode == 'auto' and 'acc' not in self.monitor)):
                 (self.mode == 'auto')):
             self.monitor_op = lambda a, b: less(a, b - self.min_delta)
             self.best = Inf
         else:
             self.monitor_op = lambda a, b: greater(a, b + self.min_delta)
             self.best = -Inf
         self.cooldown_counter = 0
         self.wait = 0

    def on_train_begin(self, logs=None):
      self._reset()
    
    def on_epoch_end(self, epoch, loss, logs=None):
    
    
        logs = logs or {}
        ## Custom modification: Optimizer
        # logs['lr'] = K.get_value(self.model.optimizer.lr) returns a numpy array
        # and therefore can be modified to          
        logs['lr'] = float(self.optim_lr.numpy())
    
        ## Custom modification: Deprecated due to focusing on validation loss
        # current = logs.get(self.monitor)
    
        current = float(loss)
    
        ## Custom modification: Deprecated due to focusing on validation loss
        # if current is None:
        #     print('Reduce LR on plateau conditioned on metric `%s` '
        #                     'which is not available. Available metrics are: %s',
        #                     self.monitor, ','.join(list(logs.keys())))
    
        # else:
    
        if self.in_cooldown():
            self.cooldown_counter -= 1
            self.wait = 0
    
        if self.monitor_op(current, self.best):
            self.best = current
            self.wait = 0
        elif not self.in_cooldown():
            self.wait += 1
            if self.wait >= self.patience:
    
                ## Custom modification: Optimizer Learning Rate
                # old_lr = float(K.get_value(self.model.optimizer.lr))
                old_lr = float(self.optim_lr.numpy())
                if old_lr > self.min_lr and self.reduce_lr == True:
                    ## Custom modification: Linear learning Rate
                    if self.reduce_lin == True:
                        new_lr = old_lr * self.factor
                        ## Custom modification: Error Handling when learning rate is below zero
                        if new_lr <= 0:
                            print('Learning Rate is below zero: {}, '
                            'fallback to minimal learning rate: {}. '
                            'Stop reducing learning rate during training.'.format(new_lr, self.min_lr))  
                            self.reduce_lr = False                           
                    else:
                        new_lr = old_lr * self.factor                   
    
    
                    new_lr = max(new_lr, self.min_lr)
    
    
                    ## Custom modification: Optimizer Learning Rate
                    # K.set_value(self.model.optimizer.lr, new_lr)
                    self.optim_lr.assign(new_lr)
    
                    if self.verbose > 0:
                        print('\nEpoch %05d: ReduceLROnPlateau reducing learning '
                                'rate to %s.' % (epoch + 1, float(new_lr)))
                    self.cooldown_counter = self.cooldown
                    self.wait = 0
    
    def in_cooldown(self):
      return self.cooldown_counter > 0

def tf_create_angles(nproj, scan = '180'):
    
    '''
    Create the projection angles
    '''
    
    if scan=='180':
        theta = np.arange(0, 180, 180/nproj)
    elif scan=='360':
        theta = np.arange(0, 360, 360/nproj)
        
    theta_tf = tf.convert_to_tensor(np.radians(theta), dtype=tf.float32)
    
    return(theta_tf)

def tf_tomo_transf(im):
    return(tf.transpose(tf.reshape(im, (im.shape[0], im.shape[1], 1, 1)), (2, 0, 1, 3)))

def tf_tomo_radon(rec, ang, tile = True, norm = False, interp_method = 'bilinear'):
    
    '''
    Create the radon transform of an image
    Inputs:
        rec: 4D array corresponding to (1, npix, npix, 1)
        ang: 1D array corresponding to the projection angles
    '''
    
    nang = ang.shape[0]
    img = tf.transpose(rec, [3, 1, 2, 0])
    
    if tile == True:
        img = tf.tile(img, [nang, 1, 1, 1])
        img = tfa.image.rotate(img, -ang, interpolation = interp_method)
        sino = tf.reduce_sum(img, 1, name=None)
        sino = tf.transpose(sino, [2, 0, 1])
        sino = tf.reshape(sino, [sino.shape[0], sino.shape[1], sino.shape[2], 1])
    else:
        sino = tf.zeros((0, img.shape[1], 1))
        for ii in range(nang):
            sino = tf.concat([sino,tf.reduce_sum(tfa.image.rotate(img, -ang[ii], interpolation = interp_method), 1)], 0)
        sino = tf.reshape(sino, [1, sino.shape[0], sino.shape[1], sino.shape[2]])
        
    if norm == True:
        sino = tf.image.per_image_standardization(sino)
    return sino

def tf_tomo_squeeze(im):
    return(im[0,:,:,0])

def tf_tomo_bp(sino, ang, projmean = False, norm = False, interp_method = 'bilinear'):
    
    '''
    Create the CT back projected image
    Inputs:
        sino: 4D array corresponding to (1, nproj, npix, 1)
        ang: 1D array corresponding to the projection angles
    '''
    d_tmp = sino.shape
    prj = tf.reshape(sino, [1, d_tmp[1], d_tmp[2], 1])
    prj = tf.tile(prj, [d_tmp[2], 1, 1, 1])
    prj = tf.transpose(prj, [1, 0, 2, 3])
    prj = tfa.image.rotate(prj, ang, interpolation = interp_method)
    
    if projmean == True:
        bp = tf.reduce_mean(prj, 0)
    else:
        bp = tf.reduce_sum(prj, 0) * tf.convert_to_tensor(np.pi / (len(ang)), dtype='float32')
    
    if norm == True:
        bp = tf.image.per_image_standardization(bp)
    bp = tf.reshape(bp, [1, bp.shape[0], bp.shape[1], 1])
    return bp

def tf_mask_circle(img, npix=0):    
    # Mask everything outside the reconstruction circle
    sz = tf.math.floor(float(img.shape[1]))
    x = tf.range(0,sz)
    x = tf.repeat(x, int(sz))
    x = tf.reshape(x, (sz, sz))
    y = tf.transpose(x)
    xc = tf.math.round(sz/2)
    yc = tf.math.round(sz/2)
    r = tf.math.sqrt(((x-xc)**2 + (y-yc)**2))
    img = tf.where(r>sz/2 - npix,0.0,img[:,:,:,0])
    img = tf.reshape(img, (1, sz, sz, 1))
    return(img)

def ssim_mae_loss(y_true, y_pred):
    return((1-0.84)*tf.reduce_mean(tf.keras.losses.MAE(y_pred, y_true)) + 0.84*(1 - tf.reduce_mean(tf.image.ssim(y_pred, y_true, 2.0))))

def discriminator_loss(loss_object, real_output, fake_output):
    real_loss = loss_object(tf.ones_like(real_output), real_output)
    fake_loss = loss_object(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss