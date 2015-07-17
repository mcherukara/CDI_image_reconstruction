__author__ = 'mathew'

import tifffile as tff
import numpy as np
import params
import methods as meth
import time
import pyfftw.interfaces.scipy_fftpack as pf
import pyfftw as pyf
import Tools as t
import matplotlib.pyplot as plt


"""Read TIFF stack and store as 3D numpy array"""
im=tff.TiffFile(params.ifile) #Need full path
imarr=im.asarray()

"""Get dimensions of data"""
nz,nx,ny=imarr.shape[0],imarr.shape[1],imarr.shape[2]
print "Original number of Z,X and Y bins", nz,nx,ny

"""Subtracting background comes here"""




"""FFTW parameters and error storage"""
pyf.interfaces.cache.enable() #Enable cache to store instances of pf.FFTW class, creating overhead is large
pyf.interfaces.cache.set_keepalive_time(60) #How long to keep cache alive in seconds
err=[]


"""Threshold values"""
imarr=np.where(imarr<params.thresh,0,imarr)
#print imarr.dtype
tff.imsave(params.threshed_file,imarr)

"""Sqrt to get amplitudes"""
imarr=imarr**0.5
#Create larger array to hold centred data
data=np.zeros((2*nz,2*nx,2*ny),imarr.dtype)

""" Find position of max intensity and center data there"""
maxpos=np.unravel_index(imarr.argmax(),imarr.shape) #Gives you index of max as a tuple
shift_z=nz-maxpos[0]
shift_x=nx-maxpos[1]
shift_y=ny-maxpos[2]
#print "Shifts", shift_z, shift_x, shift_y
data[shift_z:shift_z+nz,shift_x:shift_x+nx,shift_y:shift_y+ny]=imarr.copy()
print "New bounds", shift_z,shift_z+nz,shift_x,shift_x+nx,shift_y,shift_y+ny
#print data.dtype
tff.imsave(params.cent,data)



"""Build support"""
sz_pos,sz_neg=params.zf*nz/2,(-1)*params.zf*nz/2
sx_pos,sx_neg=params.xf*nx/2,(-1)*params.xf*nx/2
sy_pos,sy_neg=params.yf*ny/2,(-1)*params.yf*ny/2
supportparams=((sz_pos,0,0),(sz_neg,0,0),(0,sx_pos,0),(0,sx_neg,0),(0,0,sy_pos),(0,0,sy_neg))
support=t.MakePoly(data.shape, supportparams)

"""Create initial guess and scale data"""
#iguess=np.zeros((nz,nx,ny),dtype=complex)
data=pf.fftshift(data) #ONLY shift ever done
#data=pf.fftn(pf.fftshift(pf.ifftn(pf.fftshift(data))))
dmag=np.sum(np.abs(data)**2)
#iguess=data*np.exp(1j*np.random.uniform(-np.pi/2,np.pi/2))
#guess=pf.fftshift(pf.ifftn(iguess))

guess=support*np.random.random(support.shape).astype(np.complex128)*np.exp(1j*np.random.uniform(-np.pi/2,np.pi/2))
sos2=1 #Real space scale,norm to 0 index point
sos2=np.sum(np.abs(guess)**2)
scale=data.ravel()[0]/sos2
print "Scale", scale
guess*=scale



"""Iterate over algorithms"""
print "Entering iterative loop"
a=time.clock()
i=1
while(i<params.n_iters):

     #Run first ER runs
    for k in range(params.n_ER):
        if (k==0): print "Starting ER"
        tmp=meth.ER(guess,support)
        #meth.xyz_save(guess,'visuals/ER_pre_meth.xyz',params.scalez,params.scalex,params.scaley)
        tmpa,oguess,guess=meth.each_iter(guess,tmp,data,dmag)
        err.append(tmpa)
        print "Error", i, tmpa
        i+=1
    meth.xyz_save(guess,'visuals/curr_ER.xyz',params.scalez,params.scalex,params.scaley)

    #Run prescribed HIO iters
    for k in range(params.n_HIO):
        if(k==0):print "Starting HIO"
        tmp=meth.HIO(oguess,guess,support,params.beta,params.nph,params.pph)
        #meth.xyz_save(guess,'visuals/HIO_pre_meth.xyz',params.scalez,params.scalex,params.scaley)
        tmpa,oguess,guess=meth.each_iter(guess,tmp,data,dmag)
        err.append(tmpa)
        print "Error", i, tmpa
        i+=1
    meth.xyz_save(guess,'visuals/curr_HIO.xyz',params.scalez,params.scalex,params.scaley)

    #Same for ER
    for k in range(params.n_ER):
        if (k==0): print "Starting ER"
        tmp=meth.ER(guess,support)
        #meth.xyz_save(guess,'visuals/ER_pre_meth.xyz',params.scalez,params.scalex,params.scaley)
        tmpa,oguess,guess=meth.each_iter(guess,tmp,data,dmag)
        err.append(tmpa)
        print "Error", i, tmpa
        i+=1
        #meth.xyz_save(guess,'visuals/curr_ER.xyz',params.scalez,params.scalex,params.scaley)

    #If OSS control must fall directly here
    varg=params.oss_var
    for k in range(params.n_OSS):
        if(k==0): print "Starting OSS"
        varg/=2
        tmp=meth.OSS(oguess,guess,support,params.beta,varg)
        tmpa,oguess,guess=meth.each_iter(guess,tmp,data,dmag)
        print "Error", i, tmpa
        err.append(tmpa)
        i+=1

    print "Loop",i
b=time.clock()
print "Loop time", b-a


a=time.clock()
meth.xyz_save(guess,'visuals/tst.xyz',params.scalez,params.scalex,params.scaley)
b=time.clock()
print "File write time", b-a
print "Error with iteration", err
plt.plot(err)
plt.show()