import numpy as np
from skimage.transform import iradon
from tqdm import tqdm

def fbpvol(svol, scan = 180, theta=None, nt = None):
    
    '''
    Calculates the reconstructed images of a stack of sinograms using the filtered backprojection algorithm, 3rd dimension is z/spectral
    '''
    if nt is None:
        nt = svol.shape[0]
    nproj = svol.shape[1]
    
    if theta is None:
        theta = np.arange(0, scan, scan/nproj)
    
    if len(svol.shape)>2:
    
        vol = np.zeros((nt, nt, svol.shape[2]))
        
        for ii in tqdm(range(svol.shape[2])):
            
            vol[:,:,ii] = iradon(svol[:,:,ii], theta, nt, circle = True)
                
    elif len(svol.shape)==2:
        
        vol = iradon(svol, theta, nt, circle = True)
    
    print('The dimensions of the reconstructed volume are ', vol.shape)
        
    return(vol)

def sinocentering(sinograms, crsr=5, interp=True, scan=180):
            
    """
    Method for centering sinograms by flipping the projection at 180 deg and comparing it with the one at 0 deg
    Sinogram can be a 2D or 3D matrix (stack of sinograms)
    Dimensions: translation steps (detector elements), projections, z (spectral)
    """   
    
    di = sinograms.shape
    if len(di)>2:
        s = np.sum(sinograms, axis = 2)
    else:
        s = np.copy(sinograms)
        
    if scan==360:
        
        s = s[:,0:int(np.round(s.shape[1]/2))]
    
    cr =  np.arange(s.shape[0]/2 - crsr, s.shape[0]/2 + crsr, 0.1)
    
    xold = np.arange(0,s.shape[0])
    
    st = []; ind = [];
    
    print('Calculating the COR')
    
    for kk in tqdm(range(len(cr))):
        
        xnew = cr[kk] + np.arange(-np.ceil(s.shape[0]/2), np.ceil(s.shape[0]/2)-1)
        sn = np.zeros((len(xnew),s.shape[1]), dtype='float32')
        
        
        for ii in range(s.shape[1]):
            
            if interp==True:
                
                sn[:,ii] = np.interp(xnew, xold, s[:,ii])
            else:
                
                sn[:,ii] = np.interp(xnew, xold, s[:,ii], left=0 , right=0)

        re = sn[::-1,-1]
        st.append((np.std((sn[:,0]-re)))); ind.append(kk)

    m = np.argmin(st)
    print(cr[m])
    
    xnew = cr[m] + np.arange(-np.ceil(s.shape[0]/2), np.ceil(s.shape[0]/2)-1)
    
    print('Applying the COR correction')

    if len(di)>2:
        sn = np.zeros((len(xnew), sinograms.shape[1], sinograms.shape[2]), dtype='float32')  
        for ll in tqdm(range(sinograms.shape[2])):
            for ii in range(sinograms.shape[1]):
                
                if interp==True:
                    sn[:,ii,ll] = np.interp(xnew, xold, sinograms[:,ii,ll])    
                else:
                    sn[:,ii,ll] = np.interp(xnew, xold, sinograms[:,ii,ll], left=0 , right=0) 
            
    elif len(di)==2:
        
        sn = np.zeros((len(xnew),sinograms.shape[1]), dtype='float32')    
        for ii in range(sinograms.shape[1]):
            sn[:,ii] = np.interp(xnew, xold, sinograms[:,ii], left=0 , right=0)
        
    return(sn)