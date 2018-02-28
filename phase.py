import tifffile as tff
import numpy as np
import params as p
import pyfftw.interfaces.scipy_fftpack as sf
import pyfftw as pyf
import Tools as T
import matplotlib.pyplot as plt
import math as m
import methods as meth
from pyevtk.hl import *

#Read image
im=tff.TiffFile(p.ifile)
imarr=im.asarray()
imarr=imarr.transpose(2,1,0) #I think the detector is rotated 90?
nx,ny,nz=imarr.shape[0],imarr.shape[1], imarr.shape[2]
print "Original number of X,Y and Z bins", nx, ny, nz

#Python wrapper for FFTW3
pyf.interfaces.cache.enable() #Enable cache to store instances of pf.FFTW class, creating overhead is large
pyf.interfaces.cache.set_keepalive_time(600) #How long to keep cache alive in seconds
err=[]
merr=[]

#Threshold values
imarr=np.where(imarr<p.thresh,0,imarr)
tff.imsave(p.threshed_file,imarr)

#Take square root of measured intensities
imarr=imarr**0.5

#Set size of object array
dx,dy,dz=m.floor(p.dfs*nx),m.floor(p.dfs*ny),m.floor(p.dfs*nz)
dx,dy,dz=dx+dx%2,dy+dy%2,dz+dz%2 #Numbers must be even
data=np.zeros((dx,dy,dz))

#Find position of max intensity and centre data there
maxpos=np.unravel_index(imarr.argmax(),imarr.shape) #Gives you index of max as a tuple
shift_x=dx/2-maxpos[0]
shift_y=dy/2-maxpos[1]
shift_z=dz/2-maxpos[2]
data[shift_x:shift_x+nx,shift_y:shift_y+ny,shift_z:shift_z+nz]=imarr
print "New bounds", shift_x,shift_x+nx,shift_y,shift_y+ny,shift_z,shift_z+nz
tff.imsave(p.cent,data)

data=sf.fftshift(data) #Shift data so D.C element is at 0
dmag=np.sum(np.abs(data)**2)
sos2=dmag/data.size #NOTE USE OF SIZE HERE BUT NOT AT SOS1
DC_elem=np.abs(data.ravel()[0])

#Build support
sx_pos,sx_neg=p.xf*dx/2,(-1)*p.xf*dx/2
sy_pos,sy_neg=p.yf*dy/2,(-1)*p.yf*dy/2
sz_pos,sz_neg=p.zf*dz/2,(-1)*p.zf*dz/2
supportparams=((sx_pos,0,0),(sx_neg,0,0),(0,sy_pos,0),(0,sy_neg,0),(0,0,sz_pos),(0,0,sz_neg))
support=T.MakePoly(data.shape, supportparams) #Ross's function to create a polygon using bounds

#Create initial guess
rand1=np.random.random(support.shape)-0.5
guess=support*np.exp(1j*rand1)

#Normalize guess, normalize by energy
sos1=np.sum(np.abs(guess)**2)
print "Sum of real and fourier(averaged) squares", sos1, sos2
scale=(sos2/sos1)**0.5
guess*=scale
print "Scale, D.C element, Mag of sum", scale, DC_elem, np.abs(np.sum(guess))
print "New SoS guess and Sos data", np.sum(np.abs(guess)**2), sos2

iters=0
for i in range(p.n_cycles):
 #First ER
 for k in range(p.n_ER):
   guess=meth.ER(dmag,data[:,:,:],support[:,:,:],guess[:,:,:],err,merr)
   iters+=1
   if(iters%p.severy==0 and i==0): #Update support when it is time but only do it on first cycle
     support=meth.upsup(guess[:,:,:],iters)

 #meth.xyz_save(guess,'visuals/ER.%d.xyz' %i)


 #Phase_HIO
 for k in range(p.n_HIO):
   guess=meth.HIO(dmag,data[:,:,:],support[:,:,:],guess[:,:,:],err,merr)
   iters+=1
   if(iters%p.severy==0 and i==0):
     support=meth.upsup(guess[:,:,:],iters)

 #meth.xyz_save(guess,'visuals/HIO.%d.xyz' %i)


 #Second ER
 for k in range(p.n_ER):
   guess=meth.ER(dmag,data[:,:,:],support[:,:,:],guess[:,:,:],err,merr)
   iters+=1
   if(iters%p.severy==0 and i==0):
     support=meth.upsup(guess[:,:,:],iters)


#Write data to favourite formats
meth.xyz_save(guess,'visuals/tst.xyz') #XYZ without coordiante transform

#Return to actual real space coordinates--Ross has encoded Pfeifer's implementation
dx,dy,dz=1./guess.shape[0],1./guess.shape[1],1./guess.shape[2]
dpx,dpy=p.pixelx/p.arm, p.pixely/p.arm
trans=meth.getCoordSystem(guess.shape,'direct')
intens=np.abs(guess)
phase=np.arctan2(guess.imag,guess.real)
gridToVTK("visuals/tst",trans[:,:,:,0],trans[:,:,:,1],trans[:,:,:,2],pointData={"Intensity":intens,"Phase":phase})

#Plot error progression measured as difference of squares and scaling factor
plt.figure(1)
p1=plt.subplot(211)
p1.set_ylabel("Squares error")
plt.plot(err)
p2=plt.subplot(212)
p2.set_ylabel("Scale")
p2.set_xlabel("Iteration number")
plt.plot(merr)
plt.show()
