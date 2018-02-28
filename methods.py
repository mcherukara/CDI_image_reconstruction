__author__ = 'mathew'

import cmath
import math as m
import numpy as np
import pyfftw.interfaces.scipy_fftpack as sf
import params as p
import time
import Tools as T
import scipy.ndimage.filters as nf
from scipy.spatial import ConvexHull as CH
from skimage.morphology import convex_hull_image as CHI

#ER
def ER(dmag,data,support,guess,err,merr):
  iguess=sf.fftn(guess)
  cerr=np.sum( ( np.abs(iguess)-np.abs(data) )**2 )/dmag #Calculate error
  phi=np.arctan2(iguess.imag,iguess.real)
  iguess=np.where(np.abs(data),data*np.exp(1.j*phi),0.0) #Replace with measured amplitudes
  oguess=guess
  guess=sf.ifftn(iguess)
  sos1=np.sum(np.abs(guess)**2)
  guess*=support #Update real space object
  sos2=np.sum(np.abs(guess)**2) #Rescale amplitudes after update
  scale=(sos1/sos2)**0.5
  guess*=scale
  err.append(cerr)
  merr.append(scale)
  print cerr, scale
  return guess


#Phase constrained HIO
def HIO(dmag,data,support,guess,err,merr):
  iguess=sf.fftn(guess)
  cerr=np.sum( ( np.abs(iguess)-np.abs(data) )**2 )/dmag
  phi=np.arctan2(iguess.imag,iguess.real)
  iguess=np.where(np.abs(data),data*np.exp(1.j*phi),0.0) #Replace with measured amplitudes
  oguess=guess
  guess=sf.ifftn(iguess)
  sos1=np.sum(np.abs(guess)**2)
  phr=np.arctan2(guess.imag, guess.real)
  chk=np.logical_and(support,np.logical_and((p.nph<phr),(phr<p.pph))) #Phase constraint
#  guess=support*guess+(1-support)*(oguess-p.beta*guess)
  guess=chk*guess+(1-chk)*(oguess-p.beta*guess) #Phase coonstrained HIO
  sos2=np.sum(np.abs(guess)**2) #Rescale amplitdues after update
  scale=(sos1/sos2)**0.5
  guess*=scale 
  err.append(cerr)
  merr.append(scale)
  print cerr, scale
  return guess

#Shrink wrap and update support by scaling guassian smoothening with increasing iterations
def upsup(obj,i):
  ntot=p.n_iters/p.n_cycles #Only shrink support during the first cycle
  xo,xf=1.,ntot/p.severy #Calculate initial and final update points
  yo,yf=p.isig,float(p.fsig)
  i/=float(p.severy) #Count in number of updates
  if(xf==1):sig=(p.isig+p.fsig)/2. #0 division of only 1 update done
  else: sig=yo+(i-xo)*((yf-yo)/(xf-xo)) #Linearly scale sigma between initial and final values
  print "Real space sigma", sig
  rimage=np.abs(obj)

  smooth=np.abs(T.gauss_conv_fft(rimage,[sig,sig,sig]))
  smooth/=smooth.max()
  supp=(smooth>=p.sfrac)*1 #Threshold as fraction of max
#  xyz_save(supp,'visuals/supp%d.xyz' %i)
  return supp
  

#Method to write array to XYZ for Ovito, after coordinate transform
def xyz_trans(cors,oguess,name):
    a=time.clock()
    nx,ny,nz=oguess.shape[0],oguess.shape[1],oguess.shape[2]
    nd=2
    nats=(nx//nd)*(ny//nd)*(nz//nd)
    ofile=open(name,'w')
    ofile.write("%d\n" %nats)
    ofile.write("X Y Z Intensity Phase\n")
    for i in range(0,nx,nd):
        for j in range(0,ny,nd):
            for k in range(0,nz,nd):
                tmp=cmath.polar(oguess[i,j,k])
                ofile.write("%.2f %.2f %.2f %.4f %.4f\n" %(cors[i,j,k,0],cors[i,j,k,1],cors[i,j,k,2],tmp[0],tmp[1]))
    ofile.close()
    b=time.clock()
    print "File write time", b-a
    return


#Lifted from Ross who used the transform outlined in Pfiefer's thesis (except for the 1st column)
def getCoordSystem(dims,space):

  dpx,dpy = p.pixelx/p.arm,p.pixely/p.arm
  #dQdpx=np.array([-m.cos(p.delta),0.0,m.sin(p.delta)]) #Pfeifer has this without cosine terms on his thesis!!!!!
  dQdpx=np.array([-m.cos(p.delta)*m.cos(p.gam),0.0,m.sin(p.delta)*m.cos(p.gam)])
  dQdpy=np.array([m.sin(p.delta)*m.sin(p.gam),-m.cos(p.gam),m.cos(p.delta)*m.sin(p.gam)])
  dQdth=np.array([1-m.cos(p.delta)*m.cos(p.gam),0.0,m.sin(p.delta)*m.cos(p.gam)])

  Astar=(2.0*m.pi/p.lam)*dpx*dQdpx
  Bstar=(2.0*m.pi/p.lam)*dpy*dQdpy
  Cstar=(2.0*m.pi/p.lam)*p.dth*dQdth
  denom = np.dot( Astar, np.cross(Bstar,Cstar) )
  A=2*m.pi*np.cross(Bstar,Cstar)/denom
  B=2*m.pi*np.cross(Cstar,Astar)/denom
  C=2*m.pi*np.cross(Astar,Bstar)/denom

  if p.delta >= 0.0 and p.gam >= 0.0 and p.pixelx > 0 and \
    p.pixely > 0 and p.dth >= 0.0 and \
    p.lam >= 0.0 and p.arm >= 0.0:
    if space=="direct":
      #T = nd.array((A,B,C)).transpose()
      T = np.array((A,B,C))
    elif space=="recip":
      #T = nd.array((Astar, Bstar, Cstar)).transpose()
      T = np.array((Astar, Bstar, Cstar))
  else:
    T = np.array((1,0,0,0,1,0,0,0,1), dtype=float)
  T.shape=(3,3)

  print "transform", T
  print "dims in getcoord", dims
  if len(dims) < 2: 
    "I'm not doing 2D data sets"
    return
  
  size1,size2,size3=dims[0],dims[1],dims[2]
  dx,dy,dz=1./size1,1./size2,1./size3
  r=np.mgrid[ (size1-1)*dx:-dx:-dx, 0:size2*dy:dy, 0:size3*dz:dz]
  s1,s2,s3=r.shape[1],r.shape[2],r.shape[3] #The 1st coordinate sometimes gets an extra bin
  r=r.transpose()
  coords = np.dot(r, T)
  coords=np.reshape(coords,(s1,s2,s3,3))
  print coords.shape

  return coords


#Method to write array to XYZ for Ovito, don't bother with transform
def xyz_save(oguess,name):
    a=time.clock()
    nx,ny,nz=oguess.shape[0],oguess.shape[1],oguess.shape[2]
    nd=2
    nats=(nx//nd)*(ny//nd)*(nz//nd)
    ofile=open(name,'w')
    ofile.write("%d\n" %nats)
    ofile.write("X Y Z Intensity Phase\n")
    for i in range(0,nx,nd):
        for j in range(0,ny,nd):
            for k in range(0,nz,nd):
                tmp=cmath.polar(oguess[i,j,k])
                ofile.write("%.2f %.2f %.2f %.4f %.4f\n" %(i,j,k,tmp[0],tmp[1]))
    ofile.close()
    b=time.clock()
    print "File write time", b-a
    return

