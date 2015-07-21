__author__ = 'mathew'


import scipy.signal as signal
import cmath
import numpy as np
import pyfftw.interfaces.scipy_fftpack as pf
import params as p
import time
import Tools as T

"""
def gauss3d(nx,ny,nz,varg):
#Generate a 3D Gaussian in k space using number of points in each dimension
  npts=max(nx,ny,nz)
  gx=signal.gaussian(npts,varg)
  gx/=sum(gx)
  gy=signal.gaussian(npts,varg)
  gy/=sum(gy)
  gz=signal.gaussian(npts,varg)
  gz/=sum(gz)
  gout=gx*gy*gz
  print gout.shape
  meth.xyz_save(gout,'visuals/gout.xyz',params.scalez,params.scalex,params.scaley)
  return gout
"""


def ER(eguess, support):
    a=time.clock()
    rs1=np.abs(np.sum(eguess)) #Real scale 1
    eguess*=support
    rs2=np.abs(np.sum(eguess)) #Real scale 2
    eguess*=rs1/rs2
    b=time.clock()
    print "ER time", b-a
    return eguess


def HIO(oguess,eguess,support,beta,nph,pph):
    a=time.clock()
    rs1=np.abs(np.sum(eguess)) #Real scale 1
    """Edit these to include phase constraint, in general support seems more important"""
    ph=np.arctan2(eguess.imag,eguess.real)
    chk=np.logical_and(support>0,nph<ph,ph<pph)
    nguess=np.where(chk,eguess, oguess-beta*eguess)
    rs2=np.abs(np.sum(nguess)) #Real scale 2
    nguess*=rs1/rs2
    b=time.clock()
    print "HIO time", b-a
    return nguess


def OSS(oguess,eguess,support,beta,iter):
    tmp=np.where(support,eguess, oguess-beta*eguess) #Same as HIO-no phase for now
    nz,nx,ny=eguess.shape[0],eguess.shape[1],eguess.shape[2]
    factor=iter/float(p.n_OSS)
    vx,vy,vz=nx
    T.gauss_conv_fft(tmp)
    #tmp2=pf.ifftn(pf.fftn(tmp)*gauss3d(nx,ny,nz,varg))
    return guess*support+(1-support)*tmp2


def each_iter(eguess,etmp,data,dmag,dc): #Copies previous iteration completes full circle to come back to real space
#    pyf.import_wisdom(wisdom)
    a=time.clock()
    oguess=eguess
    eguess=etmp
    iguess=pf.fftn(eguess)
    #print "Real space check", np.abs(np.sum(eguess)), np.abs(iguess.ravel()[0])
    err=np.sum((np.abs(iguess)-np.abs(data))**2)/dmag
    iguess=np.where(data>0,np.abs(data)/np.abs(iguess)*iguess,iguess) #Only replace intensities when the threshold>something--this is done at the image stage
    #iguess=np.abs(data)/np.abs(iguess)*iguess
    eguess=pf.ifftn(iguess)
    #print "Fourier space check", np.abs(iguess.ravel()[0]), np.abs(np.sum(eguess))
    b=time.clock()
    print "Iteration time", b-a
    return err,oguess,eguess


def xyz_save(oguess,name,scalex,scaley,scalez):
    a=time.clock()
    nx,ny,nz=oguess.shape[0],oguess.shape[1],oguess.shape[2]
    nats=(nx//3+1)*(ny//3+1)*(nz//3+1)
    ofile=open(name,'w')
    ofile.write("%d\n" %nats)
    ofile.write("X Y Z Intensity Phase\n")
    lcnt=0 #----RETURN TO THIS, SHOULD WRITE ONLY NEEDED REGIONS, MEMORY WILL BE AN ISSUE
    for i in range(0,nx,3):
        for j in range(0,ny,3):
            for k in range(0,nz,3):
                tmp=cmath.polar(oguess[i,j,k])
    #    if(tmp[0]>0):#If number has magnitude, plot
                ofile.write("%.2f %.2f %.2f %.4f %.4f\n" %(i*scalex,j*scaley,k*scalez,tmp[0],tmp[1]))
                lcnt+=1
    ofile.close()
    b=time.clock()
    print "File write time", b-a
  #print "%d Lines written" %lcnt
    return


