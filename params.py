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
n_iters=20 #Will run x*(n_HIO+n_ER) that carries the running total over n_iters
n_HIO=15
n_ER=5
n_OSS=0

#Initial OSS variance
oss_var=10

#Phasing threshold and image subtraction threshold
thresh=2

#Phase range
nph=-np.pi/2
pph=np.pi/2

#Feedback parameter (set between 0.7 and 0.9)
beta=0.9

#Scaling parameter for each pixel
scalex=1
scaley=1
scalez=1

#Initial box support (in fractions of half data array size)
xf=1.3
yf=1.3
zf=1.3

# Detector and scan variables
deg2rad=math.pi/180
pixelx=4*(22.5e-6)  #Total pixel size including binning (in meters)
pixely=4*(22.5e-6)
arm = .66          #Detector distance from sample (units are m)
lam = 0.13933          #Wavelength of x-ray (units in nm)
delta = 31.4 * deg2rad    #Detector angles
gam =  13.9 * deg2rad
dth = 0.020 * deg2rad      #step values for th translation
dtilt = 0.000 *deg2rad





