import numpy as np
import healpy as hp

from .pstools import whitenoise

def sim_whitenoise(nside,lmax,w,theta,ellkneeT=0,alphaT=0,ellkneeP=0,alphaP=0,lmin=2):
	nl,pTl,pPl,bl=whitenoise(lmax,w,theta,ellkneeT,alphaT,ellkneeP,alphaP,lmin=lmin)
	return hp.synfast((nl/2.*pTl*bl,nl*pPl*bl,nl*pPl*bl,np.zeros(lmax+1)),nside,lmax=lmax,new=True)[:3]
	
