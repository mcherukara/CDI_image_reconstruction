__author__ = 'mathew'

import math
import numpy as np
#Input file name
ifile='images/summededit1crop.tif'

#Thresholded image name
threshed_file='images/thresh.tif'

#Centred image
cent='images/centred.tif'

#Number of HIO, ER, OSS and cycles
n_HIO=25
n_ER=5
n_OSS=0
n_cycles=2 #Number of times to cycle ER+HIO+ER
n_iters=(2*n_ER+n_HIO+n_OSS)*n_cycles

#Shrink wrap variables (Gaussian smoothing)
isig=10 #Sigma of G for first support update
fsig=5 #Sigma for final support update
severy=5 #Update support every how many iterations?
sfrac=0.2 #Fraction of maximum intensity to use for shrink wrap boundary

#Phasing threshold and image subtraction threshold
thresh=3

#Phase range
nph=-np.pi/2
pph=np.pi/2

#Feedback parameter (set between 0.7 and 0.9)
beta=0.9

#Data array size is how many times image array?
dfs=16/10.

#Initial box support (in fractions of data array size)
xf=0.4
yf=0.4
zf=0.5

# Detector and scan variables (everything in nm)
deg2rad=math.pi/180
pixelx=1*(55e3)  #Unified units!# nm everywhere!
pixely=1*(55e3)   #-----If you bin data, include here)
arm = .635e9          #Detector distance from sample
lam = 0.13933          #Wavelength of x-ray (units in nm)
delta = 30.2 * deg2rad    #Detector angles
gam =  14.32 * deg2rad
dth = 0.010 * deg2rad      #step values for th translation
dtilt = 0.000 *deg2rad
#arm = .635          #Detector distance from sample (units are m)
#pixelx=1*(55e-6)  #Total pixel size including binning (in meters)
#pixely=1*(55e-6)
