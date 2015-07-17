__author__ = 'mathew'


import scipy.signal as signal
import cmath
import numpy as np
import pyfftw.interfaces.scipy_fftpack as pf
import methods as meth
import params as p
import time
#import pyfftw as pyf


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



def ER(eguess, support):
    a=time.clock()
    rs1=np.sum(np.abs(eguess)**2) #Real scale 1
    eguess*=support
    rs2=np.sum(np.abs(eguess)**2) #Real scale 2
    eguess*=np.sqrt(rs1/rs2)
    b=time.clock()
    print "ER time", b-a
    return eguess


def HIO(oguess,eguess,support,beta,nph,pph):
    a=time.clock()
    rs1=np.sum(np.abs(eguess)**2) #Real scale 1
    """Edit these to include phase constraint, in general support seems more important"""
    ph=np.arctan2(eguess.imag,eguess.real)
    chk=np.logical_and(support>0,nph<ph,ph<pph)
    eguess=chk*eguess+(1-chk)*(oguess-beta*eguess)
    #eguess=support*eguess+(1-support)*(oguess-beta*eguess)
    rs2=np.sum(np.abs(eguess)**2) #Real scale 2
    eguess*=np.sqrt(rs1/rs2)
    b=time.clock()
    print "HIO time", b-a
    return eguess


def OSS(oguess,guess,support,beta,varg):

    tmp=guess*support+(1-support)*(oguess-beta*guess) #Same as HIO
    sz=guess.shape
    nx,ny,nz=sz[0],sz[1],sz[2]
    tmp2=pf.ifftn(pf.fftn(tmp)*gauss3d(nx,ny,nz,varg))
    return guess*support+(1-support)*tmp2


def each_iter(eguess,etmp,data,dmag): #Copies previous iteration completes full circle to come back to real space
#    pyf.import_wisdom(wisdom)
    a=time.clock()
    oguess=eguess
    eguess=etmp
    iguess=pf.fftn(eguess)
    err=np.sum((np.abs(iguess)-np.abs(data))**2)/dmag
    #iguess=np.where(data>0,np.abs(data)*np.exp(1j*np.arctan2(iguess.imag,iguess.real)),iguess) #Only replace intensities when the threshold>something--this is done at the image stage
    iguess=np.where(data>0,np.abs(data)/np.abs(iguess)*iguess,iguess) #Only replace intensities when the threshold>something--this is done at the image stage
    eguess=pf.ifftn(iguess)
    b=time.clock()
    print "Iteration time", b-a
    return err,oguess,eguess


def xyz_save(oguess,name,scalez,scalex,scaley):
    a=time.clock()
    nz,nx,ny=oguess.shape[0],oguess.shape[1],oguess.shape[2]
    nats=(nx//3+1)*(ny//3+1)*(nz//3+1)
    #print "Writing %d data points" %(nats)
    #oguess/=np.max(np.abs(oguess)) #Scale magnitudes to 1 when writing
    ofile=open(name,'w')
    ofile.write("%d\n" %nats)
    ofile.write("X Y Z Intensity Phase\n")
    lcnt=0 #----RETURN TO THIS, SHOULD WRITE ONLY NEEDED REGIONS, MEMORY WILL BE AN ISSUE
    for i in range(0,nz,3):
        for j in range(0,nx,3):
            for k in range(0,ny,3):
                tmp=cmath.polar(oguess[i,j,k])
    #    if(tmp[0]>0):#If number has magnitude, plot
                ofile.write("%.2f %.2f %.2f %.4f %.4f\n" %(i*scalez,j*scalex,k*scaley,tmp[0],tmp[1]))
                lcnt+=1
    ofile.close()
    b=time.clock()
    print "File write time", b-a
  #print "%d Lines written" %lcnt
    return


