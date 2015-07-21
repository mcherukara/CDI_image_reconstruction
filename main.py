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
import math
from CXDVizNX import CXDViz
import tvtk

"""Read TIFF stack and store as 3D numpy array"""
im=tff.TiffFile(params.ifile) #Need full path
imarr=im.asarray()

"""Get dimensions of data"""
imarr=imarr.transpose(1,2,0)
nx,ny,nz=imarr.shape[0],imarr.shape[1],imarr.shape[2]
print "Original number of X,Y and Z bins", nx,ny,nz

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
#data=np.zeros((2*nz,2*nx,2*ny),imarr.dtype)
dfs=params.dfs
dx,dy,dz=math.floor(dfs*nx),math.floor(dfs*ny),math.floor(dfs*nz)
dx,dy,dz=dx+dx%2,dy+dy%2,dz+dz%2 #Numbers must be even
data=np.zeros((dx,dy,dz))

""" Find position of max intensity and center data there"""
maxpos=np.unravel_index(imarr.argmax(),imarr.shape) #Gives you index of max as a tuple
shift_x=dx/2-maxpos[0]
shift_y=dy/2-maxpos[1]
shift_z=dz/2-maxpos[2]
#print "Shifts", shift_z, shift_x, shift_y
data[shift_x:shift_x+nx,shift_y:shift_y+ny,shift_z:shift_z+nz]=imarr.copy()
print "New bounds", shift_x,shift_x+nx,shift_y,shift_y+ny,shift_z,shift_z+nz
#print data.dtype
tff.imsave(params.cent,data)



"""Build support"""
sx_pos,sx_neg=params.xf*dx/2,(-1)*params.xf*dx/2
sy_pos,sy_neg=params.yf*dy/2,(-1)*params.yf*dy/2
sz_pos,sz_neg=params.zf*dz/2,(-1)*params.zf*dz/2
supportparams=((sx_pos,0,0),(sx_neg,0,0),(0,sy_pos,0),(0,sy_neg,0),(0,0,sz_pos),(0,0,sz_neg))
support=t.MakePoly(data.shape, supportparams)

"""Create initial guess and scale data"""
#iguess=np.zeros((nz,nx,ny),dtype=complex)
data=pf.fftshift(data) #ONLY shift ever done
#data=pf.fftn(pf.fftshift(pf.ifftn(pf.fftshift(data))))
dmag=np.sum(np.abs(data)**2)
#iguess=data*np.exp(1j*np.random.uniform(-np.pi/2,np.pi/2))
#guess=pf.fftshift(pf.ifftn(iguess))

rand1=np.random.random(support.shape)-0.5 #Range between -0.5 and 0.5
guess=support*np.exp(1j*rand1)
sos2=1 #Real space scale,norm to 0 index point
#sos2=np.sum(np.abs(guess)**2)
sos2=np.abs(np.sum(guess))
DC_elem=data.ravel()[0]
scale=DC_elem/sos2
guess*=scale
print "Scale, D.C element, Sum of mags after scale", scale, DC_elem, np.abs(np.sum(guess))



"""Iterate over algorithms"""
print "Entering iterative loop"
a=time.clock()
i=1
while(i<params.n_iters):
     #Run first ER runs
    for k in range(params.n_ER):
        if (k==0): print "Starting ER"
        tmp=meth.ER(guess,support)
        tmpa,oguess,guess=meth.each_iter(guess,tmp,data,dmag,DC_elem)
        err.append(tmpa)
        print "Error", i, tmpa
        i+=1
    meth.xyz_save(guess,'visuals/curr_ER.xyz',params.scalez,params.scalex,params.scaley)

    #Run prescribed HIO iters
    for k in range(params.n_HIO):
        if(k==0):print "Starting HIO"
        tmp=meth.HIO(oguess,guess,support,params.beta,params.nph,params.pph)
        tmpa,oguess,guess=meth.each_iter(guess,tmp,data,dmag, DC_elem)
        err.append(tmpa)
        print "Error", i, tmpa
        i+=1
    meth.xyz_save(guess,'visuals/curr_HIO.xyz',params.scalez,params.scalex,params.scaley)

    #Same for ER
    for k in range(params.n_ER):
        if (k==0): print "Starting ER"
        tmp=meth.ER(guess,support)
        tmpa,oguess,guess=meth.each_iter(guess,tmp,data,dmag, DC_elem)
        err.append(tmpa)
        print "Error", i, tmpa
        i+=1
        #meth.xyz_save(guess,'visuals/curr_ER.xyz',params.scalez,params.scalex,params.scaley)

    #If OSS control must fall directly here
    for k in range(params.n_OSS):
        if(k==0): print "Starting OSS"
        tmp=meth.OSS(oguess,guess,support,params.beta,k)
        tmpa,oguess,guess=meth.each_iter(guess,tmp,data,dmag, DC_elem)
        print "Error", i, tmpa
        err.append(tmpa)
        i+=1

    print "Loop",i
b=time.clock()
print "Loop time", b-a


meth.xyz_save(guess,'visuals/tst.xyz',params.scalex,params.scaley,params.scalez)
print "Error with iteration", err



#viz=CXDViz()
#viz.SetArray(guess)
#space='direct'
#dx,dy,dz=1./guess.shape[0],1./guess.shape[1],1./guess.shape[2]
#viz.SetGeomCoordparams(params.lam, params.delta, params.gam, params.dpx, params.dpy, params.dth, dx, dy, dz,space)
#viz.WriteStructuredGrid("d:/temp/test.vtk")
#import mayavi as m
#v=m.mayavi()
#    v.open_vtk_data(tvtk.to_vtk(viz.GetImageData()))
#v.open_vtk_data(tvtk.to_vtk(viz.GetStructuredGrid()))
#v.master.wait_window()

plt.plot(err)
plt.show()
#trans=getCoordSystemParams(params,guess.shape,'direct')
#output=np.dot(guess,trans)
#meth.xyz_save(output,'visuals/out.xyz',params.scalex,params.scaley,params.scalez)

