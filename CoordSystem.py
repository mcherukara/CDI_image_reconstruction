

import math as m
import numpy as nd
import os

def getCoordSystemParams(pp, dims, space):
  
  return getCoordSystem(pp.lam, pp.delta, pp.gam, pp.pixelx, pp.pixely, pp.arm, pp.dth, dims, space)

def getCoordSystem(lam, delta, gam, pixelx, pixely, arm, dth, dims, space):

  deg2rad=m.pi/180.0
  oneonk=lam/(2.0*m.pi)

  dpx = pixelx/arm
  dpy = pixely/arm
  delta=delta*deg2rad
  gam=gam*deg2rad

  dQdpx=nd.array((-m.cos(delta),
                   0.0,
                   m.sin(delta)))
  dQdpy=nd.array((m.sin(delta)*m.sin(gam),
                  -m.cos(gam),
                   m.cos(delta)*m.sin(gam)))
  dQdth=nd.array((-m.cos(delta)*m.cos(gam)+1,
                    0.0,
                    m.sin(delta)*m.cos(gam)))

  Astar=(2.0*m.pi/lam)*dpx*dQdpx
  Bstar=(2.0*m.pi/lam)*dpy*dQdpy
  Cstar=(2.0*m.pi/lam)*dth*dQdth

  denom = nd.dot( Astar, nd.cross(Bstar,Cstar) )
 
  A=2*m.pi*nd.cross(Bstar,Cstar)/denom
  B=2*m.pi*nd.cross(Cstar,Astar)/denom
  C=2*m.pi*nd.cross(Astar,Bstar)/denom


  if delta >= 0.0 and gam >= 0.0 and pixelx > 0 and \
    pixely > 0 and dth >= 0.0 and \
    lam >= 0.0 and arm >= 0.0:
    if space=="direct":
      #T = nd.array((A,B,C)).transpose()
      T = nd.array((A,B,C))
    elif space=="recip":
      #T = nd.array((Astar, Bstar, Cstar)).transpose()
      T = nd.array((Astar, Bstar, Cstar))
  else:
    T = nd.array((1,0,0,0,1,0,0,0,1), dtype=float)
  T.shape=(3,3)

  print "transform", T

# print T[0,0], T[0,1], T[0,2]
# print T[1,0], T[1,1], T[1,2]
# print T[2,0], T[2,1], T[2,2]
# print "=========================="

  print "dims in getcoord", dims
  if len(dims) < 2:
    return

  if dims[0]>1:
    size1 = dims[0]
  if dims[1]>1:
    size2 = dims[1]
  if dims[2]>1:
    size3 = dims[2]
  else:
    size3 = 1
  dx=1./size1
  dy=1./size2
  dz=1./size3

  r=nd.mgrid[ (size1-1)*dx:-dx:-dx, 0:size2*dy:dy, 0:size3*dz:dz]

  r.shape=3,size1*size2*size3
  r=r.transpose()

  print r.shape
  print T.shape

  coords = nd.dot(r, T)
# print coords

  return coords

