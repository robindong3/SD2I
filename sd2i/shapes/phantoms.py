# -*- coding: utf-8 -*-
"""
Create phantoms (2D-4D) for image processing and analysis experiments

@author: Antony Vamvakeros
"""

from skimage.data import shepp_logan_phantom
from skimage.transform import rescale

# Might need to convert the functions to class methods

def SheppLogan(npix):

    '''
    Create a Shepp Logan phantom using skimage
    '''

    im = shepp_logan_phantom()
    im = rescale(im, scale=npix/im.shape[0], mode='reflect')
    return(im)

