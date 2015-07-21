__author__ = 'mathew'

import math
import numpy as np
#Input file name
ifile='images/summededit1.tif'

#Thresholded image name
threshed_file='images/thresh.tif'

#Centred image
cent='images/centred.tif'

#Number of HIO, ER, OSS and minimum iterations
#(Set HIO and ER=0 if using OSS and vice versa)
n_HIO=16
n_ER=12 #Need at least 1 ER before HIO 2 generate a old guess and a current one
n_cycles=1 #Number of times to cycle ER+HIO+ER
n_OSS=0

if(n_OSS>0): n_iters=n_OSS
else:n_iters=(2*n_ER+n_HIO)*n_cycles


#Phasing threshold and image subtraction threshold
thresh=3

#Phase range
nph=-np.pi/2
pph=np.pi/2

#Feedback parameter (set between 0.7 and 0.9)
beta=0.9

#Data array size is how many times image array?
dfs=14/10.

#Scaling parameter for each pixel
scalex=1.0
scaley=1.0
scalez=1.0

#Initial box support (in fractions of data array size)
xf=0.5
yf=0.5
zf=0.5

# Detector and scan variables
deg2rad=math.pi/180
pixelx=1*(22.5e-6)  #Total pixel size including binning (in meters)
pixely=1*(22.5e-6)
arm = .66          #Detector distance from sample (units are m)
dpx=pixelx/arm
dpy=pixely/arm
lam = 0.13933          #Wavelength of x-ray (units in nm)
delta = 30.1 * deg2rad    #Detector angles
gam =  14.0 * deg2rad
dth = 0.020 * deg2rad      #step values for th translation
dtilt = 0.000 *deg2rad





