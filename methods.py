__author__ = 'mathew'


import scipy.signal as signal
import cmath
import numpy as np
import pyfftw.interfaces.scipy_fftpack as pf
import methods as meth
import params
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



def ER(guess, support):

  guess=np.where(support>0,guess,0.0)
  return guess


def HIO(oguess,guess,support,beta,nph,pph):
  ph=np.arctan2(guess.imag,guess.real)
  chk=np.logical_and(support>0,nph<ph,ph<pph)
  guess=np.where(chk,guess,oguess-beta*guess)
  return guess


def OSS(oguess,guess,support,beta,varg):

  tmp=guess*support+(1-support)*(oguess-beta*guess) #Same as HIO
  sz=guess.shape
  nx,ny,nz=sz[0],sz[1],sz[2]
  tmp2=pf.ifftn(pf.fftn(tmp)*gauss3d(nx,ny,nz,varg))
  return guess*support+(1-support)*tmp2


def each_iter(guess,tmp,data): #Copies previous iteration completes full circle to come back to real space
#    pyf.import_wisdom(wisdom)
    oguess=guess
    guess=tmp
    iguess=pf.fftn(guess)
    iguess=np.where(data>0,data*np.exp(1j*np.arctan2(iguess.imag,iguess.real)),iguess) #Only replace intensities when the threshold>something--this is done at the image stage
    meth.xyz_save(iguess,'visuals/curr_img.xyz',params.scalez,params.scalex,params.scaley)
    guess=pf.ifftn(iguess)
    meth.xyz_save(guess,'visuals/curr_real.xyz',params.scalez,params.scalex,params.scaley)
    return oguess,guess


def xyz_save(guess,name,scalez,scalex,scaley):
  nz,nx,ny=guess.shape[0],guess.shape[1],guess.shape[2]
  nats=(nx//5+1)*(ny//5+1)*(nz//5+1)
  #print "Writing %d data points" %(nats)
  ofile=open(name,'w')
  ofile.write("%d\n" %nats)
  ofile.write("X Y Z Intensity Phase\n")
  lcnt=0 #----RETURN TO THIS, SHOULD WRITE ONLY NEEDED REGIONS, MEMORY WILL BE AN ISSUE
  for i in range(0,nz,5):
    for j in range(0,nx,5):
      for k in range(0,ny,5):
          tmp=cmath.polar(guess[i,j,k])
    #    if(tmp[0]>0):#If number has magnitude, plot
          ofile.write("%.2f %.2f %.2f %.4f %.4f\n" %(i*scalez,j*scalex,k*scaley,tmp[0],tmp[1]))
          lcnt+=1
  ofile.close()
  #print "%d Lines written" %lcnt
  return

