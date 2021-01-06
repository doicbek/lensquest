# coding: utf-8

import numpy as np
cimport numpy as np
from libcpp.string cimport string
from libc.math cimport sqrt, floor, fabs
cimport libc
import os
import cython
from healpy import map2alm, npix2nside, almxfl, nside2npix
from healpy.pixelfunc import maptype, get_min_valid_nside
from healpy.sphtfunc import Alm as almpy
from libcpp cimport bool as cbool
from lensquest.pstools import getweights
import copy

from _common cimport tsize, arr, xcomplex, Healpix_Ordering_Scheme, RING, NEST, Healpix_Map, Alm, PowSpec, ndarray2cl1, ndarray2cl4, ndarray2map, ndarray2alm
		
def cltype(cl):
	if type(cl)==np.ndarray:
		if cl.ndim==1:
			return 1
		else:
			return np.shape(cl)[0]
	elif type(cl) is tuple or list:
		return len(cl)
	else:
		return 0
		
def t2i(t):
	if t=="T": return 0
	elif t=="E": return 1
	elif t=="B": return 2
							   
cdef extern from "_lensquest_cxx.cpp":
	cdef void est_grad(Alm[xcomplex[double]] &alm1, Alm[xcomplex[double]] &alm2, string stype, Alm[xcomplex[double]] &almP, Alm[xcomplex[double]] &almC, PowSpec &wcl, PowSpec &dcl, int lmin, int lminCMB1, int lminCMB2, int lmaxCMB1, int lmaxCMB2, int nside)
	cdef void est_amp(Alm[xcomplex[double]] &alm1, Alm[xcomplex[double]] &alm2, string stype, Alm[xcomplex[double]] &almP, PowSpec &wcl, PowSpec &dcl, int lmin, int lminCMB1, int lminCMB2, int lmaxCMB1, int lmaxCMB2, int nside)
	cdef void est_rot(Alm[xcomplex[double]] &alm1, Alm[xcomplex[double]] &alm2, string stype, Alm[xcomplex[double]] &almP, PowSpec &wcl, PowSpec &dcl, int lmin, int lminCMB1, int lminCMB2, int lmaxCMB1, int lmaxCMB2, int nside)
	cdef void est_spinflip(Alm[xcomplex[double]] &alm1, Alm[xcomplex[double]] &alm2, string stype, Alm[xcomplex[double]] &almP, Alm[xcomplex[double]] &almC, PowSpec &wcl, PowSpec &dcl, int lmin, int lminCMB1, int lminCMB2, int lmaxCMB1, int lmaxCMB2, int nside)
	cdef void est_monoleak(Alm[xcomplex[double]] &alm1, Alm[xcomplex[double]] &alm2, string stype, Alm[xcomplex[double]] &almP, Alm[xcomplex[double]] &almC, PowSpec &wcl, PowSpec &dcl, int lmin, int lminCMB1, int lminCMB2, int lmaxCMB1, int lmaxCMB2, int nside)
	cdef void est_phodir(Alm[xcomplex[double]] &alm1, Alm[xcomplex[double]] &alm2, string stype, Alm[xcomplex[double]] &almP, Alm[xcomplex[double]] &almC, PowSpec &wcl, PowSpec &dcl, int lmin, int lminCMB1, int lminCMB2, int lmaxCMB1, int lmaxCMB2, int nside)
	cdef void est_dipleak(Alm[xcomplex[double]] &alm1, Alm[xcomplex[double]] &alm2, string stype, Alm[xcomplex[double]] &almP, Alm[xcomplex[double]] &almC, PowSpec &wcl, PowSpec &dcl, int lmin, int lminCMB1, int lminCMB2, int lmaxCMB1, int lmaxCMB2, int nside)
	cdef void est_quadleak(Alm[xcomplex[double]] &alm1, Alm[xcomplex[double]] &alm2, string stype, Alm[xcomplex[double]] &almP, Alm[xcomplex[double]] &almC, PowSpec &wcl, PowSpec &dcl, int lmin, int lminCMB1, int lminCMB2, int lmaxCMB1, int lmaxCMB2, int nside)
	cdef void est_mask(Alm[xcomplex[double]] &alm1, Alm[xcomplex[double]] &alm2, string stype, Alm[xcomplex[double]] &almM, PowSpec &wcl, PowSpec &dcl, int lmin, int lminCMB1, int lminCMB2, int lmaxCMB1, int lmaxCMB2, int nside)
	cdef void map2alm_spin_iter(Healpix_Map[double] &mapQ, Healpix_Map[double] &mapU, Alm[xcomplex[double]] &almG, Alm[xcomplex[double]] &almC, int spin, int num_iter);
	cdef void alm2map_spin(Alm[xcomplex[double]] &almG, Alm[xcomplex[double]] &almC, Healpix_Map[double] &mapQ, Healpix_Map[double] &mapU, int spin);
	cdef void est_noise(Alm[xcomplex[double]] &alm1, Alm[xcomplex[double]] &alm2, string stype, Alm[xcomplex[double]] &almN, PowSpec &wcl, PowSpec &dcl, int lmin, int lminCMB, int nside)
	cdef void btemp(Alm[xcomplex[double]] &almB, Alm[xcomplex[double]] &almE, Alm[xcomplex[double]] &almP, int lminB, int lminE, int lminP, int nside)

class quest:
	def __init__(self, maps, wcl, dcl, lmin=2, lmax=None, lminCMB=2, lmaxCMB=None, lminCMB2=None, lmaxCMB2=None, map2=None, nside=None):
		self.wcl=wcl
		self.dcl=dcl
				
		nspec=cltype(self.wcl)		
				
		self.lmin=lmin
		self.lminCMB=lminCMB
		
		if lmaxCMB is None:
			if nspec==1:
				self.lmaxCMB=len(self.wcl)-1
			else:
				self.lmaxCMB=len(self.wcl[0])-1
		else:
			self.lmaxCMB=lmaxCMB
		if lmax is None:
			self.lmax=self.lmaxCMB
		else:
			self.lmax=lmax
			
		if lminCMB2 is None:
			self.lminCMB2=self.lminCMB
		else:
			self.lminCMB2=lminCMB2
		if lmaxCMB2 is None:
			self.lmaxCMB2=self.lmaxCMB
		else:
			self.lmaxCMB2=lmaxCMB2
		
		try:
			nmaps=maptype(maps)
		except TypeError:
			nmaps=-1
			
			
		if nmaps==-1:
			lmmap=almpy.getidx(almpy.getlmax(len(maps[0])),*almpy.getlm(self.lmaxCMB,np.arange(almpy.getsize(self.lmaxCMB))))
			self.alms=np.zeros((3,almpy.getsize(self.lmaxCMB)),dtype=complex)
			for i in range(len(maps)):
				self.alms[i]=maps[i][lmmap]
			if nside is None: self.nside=get_min_valid_nside(12*(2*self.lmaxCMB)**2)
			else: self.nside=nside
			if map2 is not None:
				self.alms2=np.zeros((3,almpy.getsize(self.lmaxCMB)),dtype=complex)
				for i in range(len(map2)):
					self.alms2[i]=map2[i][lmmap]
		elif nmaps==0:
			self.alms=[map2alm(maps,lmax=np.max([self.lmaxCMB,self.lmaxCMB2]))]
			if nside is None: self.nside = npix2nside(maps.size)
			else: self.nside=nside
			if map2 is not None:
				self.alms2=[map2alm(map2,lmax=np.max([self.lmaxCMB,self.lmaxCMB2]))]
		else: 
			self.alms=map2alm(maps,lmax=np.max([self.lmaxCMB,self.lmaxCMB2]))
			if nside is None: self.nside = npix2nside(maps[0].size)
			else: self.nside=nside
			if map2 is not None:
				self.alms2=map2alm(map2,lmax=np.max([self.lmaxCMB,self.lmaxCMB2]))
				
		if map2 is None:
			self.alms2=None
				
		self.queststorage={}
			
	def grad(self,type,norm=None,store=False):
		almgrad=quest_grad(self.alms,self.wcl,self.dcl,type,self.nside,self.lmin,self.lmax,self.lminCMB,self.lmaxCMB,lminCMB2=self.lminCMB2,lmaxCMB2=self.lmaxCMB2,alms2=self.alms2)
		if norm is not None:
			almgrad=almxfl(almgrad,norm)
		if store:
			if "grad" not in self.queststorage.keys():
				self.queststorage["grad"]={}
			self.queststorage["grad"][type]=almgrad
		return almgrad
        
	def amp(self,type,norm=None,store=False):
		almamp=quest_amp(self.alms,self.wcl,self.dcl,type,self.nside,self.lmin,self.lmax,self.lminCMB,self.lmaxCMB,lminCMB2=self.lminCMB2,lmaxCMB2=self.lmaxCMB2,alms2=self.alms2)
		if norm is not None:
			almamp=almxfl(almamp,norm)
		if store:
			if "amplitude" not in self.queststorage.keys():
				self.queststorage["amplitude"]={}
			self.queststorage["amplitude"][type]=almamp
		return almamp
        
	def rot(self,type,norm=None,store=False):
		almrot=quest_rot(self.alms,self.wcl,self.dcl,type,self.nside,self.lmin,self.lmax,self.lminCMB,self.lmaxCMB,lminCMB2=self.lminCMB2,lmaxCMB2=self.lmaxCMB2,alms2=self.alms2)
		if norm is not None:
			almrot=almxfl(almrot,norm)
		if store:
			if "rotation" not in self.queststorage.keys():
				self.queststorage["rotation"]={}
			self.queststorage["rotation"][type]=almrot
		return almrot
        
	def spinflip(self,type,norm=None,store=False):
		almspinflip=quest_spinflip(self.alms,self.wcl,self.dcl,type,self.nside,self.lmin,self.lmax,self.lminCMB,self.lmaxCMB,lminCMB2=self.lminCMB2,lmaxCMB2=self.lmaxCMB2,alms2=self.alms2)
		if norm is not None:
			almspinflip=almxfl(almspinflip,norm)
		if store:
			if "spinflip" not in self.queststorage.keys():
				self.queststorage["spinflip"]={}
			self.queststorage["spinflip"][type]=almspinflip
		return almspinflip
	
	def monoleak(self,type,norm=None,store=False):
		almmonoleak=quest_monoleak(self.alms,self.wcl,self.dcl,type,self.nside,self.lmin,self.lmax,self.lminCMB,self.lmaxCMB,lminCMB2=self.lminCMB2,lmaxCMB2=self.lmaxCMB2,alms2=self.alms2)
		if norm is not None:
			almmonoleak=almxfl(almmonoleak,norm)
		if store:
			if "monoleak" not in self.queststorage.keys():
				self.queststorage["monoleak"]={}
			self.queststorage["monoleak"][type]=almmonoleak
		return almmonoleak
	
	def photondir(self,type,norm=None,store=False):
		almphotondir=quest_photondir(self.alms,self.wcl,self.dcl,type,self.nside,self.lmin,self.lmax,self.lminCMB,self.lmaxCMB,lminCMB2=self.lminCMB2,lmaxCMB2=self.lmaxCMB2,alms2=self.alms2)
		if norm is not None:
			almphotondir=almxfl(almphotondir,norm)
		if store:
			if "photondir" not in self.queststorage.keys():
				self.queststorage["photondir"]={}
			self.almphotondir["photondir"][type]=almphotondir
		return almphotondir
	
	def dipleak(self,type,norm=None,store=False):
		almdipleak=quest_dipleak(self.alms,self.wcl,self.dcl,type,self.nside,self.lmin,self.lmax,self.lminCMB,self.lmaxCMB,lminCMB2=self.lminCMB2,lmaxCMB2=self.lmaxCMB2,alms2=self.alms2)
		if norm is not None:
			almdipleak=almxfl(almdipleak,norm)
		if store:
			if "dipleak" not in self.queststorage.keys():
				self.queststorage["dipleak"]={}
			self.almphotondir["dipleak"][type]=almdipleak
		return almdipleak
	
	def quadleak(self,type,norm=None,store=False):
		almquadleak=quest_quadleak(self.alms,self.wcl,self.dcl,type,self.nside,self.lmin,self.lmax,self.lminCMB,self.lmaxCMB,lminCMB2=self.lminCMB2,lmaxCMB2=self.lmaxCMB2,alms2=self.alms2)
		if norm is not None:
			almquadleak=almxfl(almquadleak,norm)
		if store:
			if "quadleak" not in self.queststorage.keys():
				self.queststorage["quadleak"]={}
			self.almphotondir["quadleak"][type]=almquadleak
		return almquadleak
	
	def mask(self,type,norm=None,store=False):
		almmask=quest_mask(self.alms,self.wcl,self.dcl,type,self.nside,self.lmin,self.lmax,self.lminCMB,self.lmaxCMB)
		if norm is not None:
			almmask=almxfl(almmask,norm)
		if store:
			if "mask" not in self.queststorage.keys():
				self.queststorage["mask"]={}
			self.queststorage["mask"][type]=almmask
		return almmask
	
	def noise(self,type,norm=None,store=False):
		almnoise=quest_noise(self.alms,self.wcl,self.dcl,type,self.nside,self.lmin,self.lmax,self.lminCMB,self.lmaxCMB)
		if norm is not None:
			almnoise=almxfl(almnoise,norm)
		if store:
			if "noise" not in self.queststorage.keys():
				self.queststorage["noise"]={}
			self.queststorage["noise"][type]=almnoise
		return almnoise
		
	def clear_memory(self):
		self.queststorage.clear()
		
	def output(self):
		return self.queststorage
		
	def make_minvariance(self,n0):
		spectra=self.queststorage["grad"].keys()
		
		nmv,weight=getweights(n0,spectra)
		
		almMV=0
		for spec in spectra:
			almMV+=almxfl(self.queststorage["grad"][spec],weight[spec])
		
		self.queststorage["grad"]["MV"]=almMV
		
	def make_biashardened(self,Rinv):
		for est in self.queststorage["grad-bh"].keys():
			self.queststorage["grad-bh"][est]=almxfl(self.queststorage["grad"][est],Rinv[est][:,0,0])
			for ik,k in enumerate(list(set(self.queststorage["grad"].keys()).intersection(["mask","noise"]))):
				self.queststorage["grad-bh"][est]+=almxfl(self.queststorage[k][est],Rinv[est][:,0,ik+1])
		
	def subtract_meanfield(self,key,meanfield):
		if key+"-mf" not in self.queststorage.keys():
			self.queststorage[key+"-mf"]={}
		for k in meanfield.keys():
			self.queststorage[key+"-mf"][k]=np.subtract(self.queststorage[key+"-mf"][k],meanfield[k])
			
	def load_in_storage(self,inp):
		for k in inp.keys():
			if k in ["grad","grad-mf","grad-bh","mask","noise"]:
				self.queststorage[k].clean()
				self.queststorage[k]=inp[k]
		
	def update_wcl(self,wcl):
		self.wcl=wcl	
		
	def update_dcl(self,dcl):
		self.dcl=dcl
		
def quest_grad(alms, wcl, dcl, type, nside, lmin, lmax, lminCMB, lmaxCMB, lminCMB2=None, lmaxCMB2=None, alms2=None):
	"""Estimate phi from a set of 2 spinned alm
	Parameters
	---------- 
	alms : list of 1, 2 or 3 arrays
	  T only, Q and U or T,Q and U maps in one list
	wcl : array-like, shape (1,lmaxCMB) or (4, lmaxCMB)
	  The input power spectra (lensed or unlensed) used in the weights of the normalization a la Okamoto & Hu, either TT only or TT, EE, BB and TE (polarization).
	dcl : array-like, shape (1,lmaxCMB) or (4, lmaxCMB)
	  The (noisy) input power spectra used in the denominators of the normalization (i.e. Wiener filtering), either TT only or TT, EE, BB and TE (polarization).
	type : string
		TT, TE, EE, TB, EB or BB
	lmin : int, scalar, optional
	  Minimum l of the normalization. Default: 2
	lmax : int, scalar, optional
	  Maximum l of the normalization. Default: lmaxCMB
	lminCMB : int, scalar, optional
	  Minimum l of the CMB power spectra. Default: 2
	lmaxCMB : int, scalar, optional
	  Maximum l of the CMB power spectra. Default: given by input cl arrays
		  
	Returns
	-------
	almP : array
		array containing the estimated lensing potential in harmonic space, unnormalized
	"""
	
	nspec=cltype(wcl)
	cdef int lmin_, lmax_, lminCMB_, lmaxCMB_,lminCMB1_, lmaxCMB1_,lminCMB2_, lmaxCMB2_, nside_
	cdef string type_
	type_=type.encode()
	
	lmin_=lmin
	lminCMB1_=lminCMB
	lminCMB2_=lminCMB2
	
	lmax_=lmax	
	lmaxCMB1_=lmaxCMB
	lmaxCMB2_=lmaxCMB2
	
	lmaxCMB_=np.max([lmaxCMB,lmaxCMB2])
	
	nside_=nside
		
	if nspec==1:
		wcl_c = np.ascontiguousarray(wcl[:lmaxCMB_+1], dtype=np.float64)
		dcl_c = np.ascontiguousarray(dcl[:lmaxCMB_+1], dtype=np.float64)
	elif nspec==4:
		wcl_c = [np.ascontiguousarray(cl[:lmaxCMB_+1], dtype=np.float64) for cl in wcl]
		dcl_c = [np.ascontiguousarray(cl[:lmaxCMB_+1], dtype=np.float64) for cl in dcl]

	if nspec==1:
		wcl_=ndarray2cl1(wcl_c, lmaxCMB_)
		dcl_=ndarray2cl1(dcl_c, lmaxCMB_)
	elif nspec==4:
		wcl_=ndarray2cl4(wcl_c[0], wcl_c[1], wcl_c[2], wcl_c[3], lmaxCMB_)
		dcl_=ndarray2cl4(dcl_c[0], dcl_c[1], dcl_c[2], dcl_c[3], lmaxCMB_)

	alms_cA = np.ascontiguousarray(alms[t2i(type[0])], dtype=np.complex128)
	if alms2 is None:
		alms_cB = np.ascontiguousarray(alms[t2i(type[1])], dtype=np.complex128)
	else:
		alms_cB = np.ascontiguousarray(alms2[t2i(type[1])], dtype=np.complex128)

	A = ndarray2alm(alms_cA, lmaxCMB_, lmaxCMB_)
	B = ndarray2alm(alms_cB, lmaxCMB_, lmaxCMB_)

	n_alm = alm_getn(lmax_, lmax_)
	almP = np.empty(n_alm, dtype=np.complex128)
	almC = np.empty(n_alm, dtype=np.complex128)
	AP = ndarray2alm(almP, lmax_, lmax_)
	AC = ndarray2alm(almC, lmax_, lmax_)

	est_grad(A[0], B[0], type_, AP[0], AC[0], wcl_[0], dcl_[0], lmin_, lminCMB1_, lminCMB2_, lmaxCMB1_, lmaxCMB2_, nside_)

	del A, B, AP, AC, wcl_, dcl_
	
	return almP, almC
    
def quest_amp(alms, wcl, dcl, type, nside, lmin, lmax, lminCMB, lmaxCMB, lminCMB2=None, lmaxCMB2=None, alms2=None):
	"""Estimate phi from a set of 2 spinned alm
	Parameters
	---------- 
	alms : list of 1, 2 or 3 arrays
	  T only, Q and U or T,Q and U maps in one list
	wcl : array-like, shape (1,lmaxCMB) or (4, lmaxCMB)
	  The input power spectra (lensed or unlensed) used in the weights of the normalization a la Okamoto & Hu, either TT only or TT, EE, BB and TE (polarization).
	dcl : array-like, shape (1,lmaxCMB) or (4, lmaxCMB)
	  The (noisy) input power spectra used in the denominators of the normalization (i.e. Wiener filtering), either TT only or TT, EE, BB and TE (polarization).
	type : string
		TT, TE, EE, TB, EB or BB
	lmin : int, scalar, optional
	  Minimum l of the normalization. Default: 2
	lmax : int, scalar, optional
	  Maximum l of the normalization. Default: lmaxCMB
	lminCMB : int, scalar, optional
	  Minimum l of the CMB power spectra. Default: 2
	lmaxCMB : int, scalar, optional
	  Maximum l of the CMB power spectra. Default: given by input cl arrays
		  
	Returns
	-------
	almP : array
		array containing the estimated lensing potential in harmonic space, unnormalized
	"""
	
	nspec=cltype(wcl)
	cdef int lmin_, lmax_, lminCMB_, lmaxCMB_,lminCMB1_, lmaxCMB1_,lminCMB2_, lmaxCMB2_, nside_
	cdef string type_
	type_=type.encode()
	
	lmin_=lmin
	lminCMB1_=lminCMB
	lminCMB2_=lminCMB2
	
	lmax_=lmax	
	lmaxCMB1_=lmaxCMB
	lmaxCMB2_=lmaxCMB2
	
	lmaxCMB_=np.max([lmaxCMB,lmaxCMB2])
	
	nside_=nside
		
	if nspec==1:
		wcl_c = np.ascontiguousarray(wcl[:lmaxCMB_+1], dtype=np.float64)
		dcl_c = np.ascontiguousarray(dcl[:lmaxCMB_+1], dtype=np.float64)
	elif nspec==4:
		wcl_c = [np.ascontiguousarray(cl[:lmaxCMB_+1], dtype=np.float64) for cl in wcl]
		dcl_c = [np.ascontiguousarray(cl[:lmaxCMB_+1], dtype=np.float64) for cl in dcl]

	if nspec==1:
		wcl_=ndarray2cl1(wcl_c, lmaxCMB_)
		dcl_=ndarray2cl1(dcl_c, lmaxCMB_)
	elif nspec==4:
		wcl_=ndarray2cl4(wcl_c[0], wcl_c[1], wcl_c[2], wcl_c[3], lmaxCMB_)
		dcl_=ndarray2cl4(dcl_c[0], dcl_c[1], dcl_c[2], dcl_c[3], lmaxCMB_)

	alms_cA = np.ascontiguousarray(alms[t2i(type[0])], dtype=np.complex128)
	if alms2 is None:
		alms_cB = np.ascontiguousarray(alms[t2i(type[1])], dtype=np.complex128)
	else:
		alms_cB = np.ascontiguousarray(alms2[t2i(type[1])], dtype=np.complex128)

	A = ndarray2alm(alms_cA, lmaxCMB_, lmaxCMB_)
	B = ndarray2alm(alms_cB, lmaxCMB_, lmaxCMB_)

	n_alm = alm_getn(lmax_, lmax_)
	almP = np.empty(n_alm, dtype=np.complex128)
	AP = ndarray2alm(almP, lmax_, lmax_)

	est_amp(A[0], B[0], type_, AP[0], wcl_[0], dcl_[0], lmin_, lminCMB1_, lminCMB2_, lmaxCMB1_, lmaxCMB2_, nside_)

	del A, B, AP, wcl_, dcl_
	
	return almP
    
def quest_rot(alms, wcl, dcl, type, nside, lmin, lmax, lminCMB, lmaxCMB, lminCMB2=None, lmaxCMB2=None, alms2=None):
	"""Estimate phi from a set of 2 spinned alm
	Parameters
	---------- 
	alms : list of 1, 2 or 3 arrays
	  T only, Q and U or T,Q and U maps in one list
	wcl : array-like, shape (1,lmaxCMB) or (4, lmaxCMB)
	  The input power spectra (lensed or unlensed) used in the weights of the normalization a la Okamoto & Hu, either TT only or TT, EE, BB and TE (polarization).
	dcl : array-like, shape (1,lmaxCMB) or (4, lmaxCMB)
	  The (noisy) input power spectra used in the denominators of the normalization (i.e. Wiener filtering), either TT only or TT, EE, BB and TE (polarization).
	type : string
		TT, TE, EE, TB, EB or BB
	lmin : int, scalar, optional
	  Minimum l of the normalization. Default: 2
	lmax : int, scalar, optional
	  Maximum l of the normalization. Default: lmaxCMB
	lminCMB : int, scalar, optional
	  Minimum l of the CMB power spectra. Default: 2
	lmaxCMB : int, scalar, optional
	  Maximum l of the CMB power spectra. Default: given by input cl arrays
		  
	Returns
	-------
	almP : array
		array containing the estimated lensing potential in harmonic space, unnormalized
	"""
	
	nspec=cltype(wcl)
	cdef int lmin_, lmax_, lminCMB_, lmaxCMB_,lminCMB1_, lmaxCMB1_,lminCMB2_, lmaxCMB2_, nside_
	cdef string type_
	type_=type.encode()
	
	lmin_=lmin
	lminCMB1_=lminCMB
	lminCMB2_=lminCMB2
	
	lmax_=lmax	
	lmaxCMB1_=lmaxCMB
	lmaxCMB2_=lmaxCMB2
	
	lmaxCMB_=np.max([lmaxCMB,lmaxCMB2])
	
	nside_=nside
		
	if nspec==1:
		wcl_c = np.ascontiguousarray(wcl[:lmaxCMB_+1], dtype=np.float64)
		dcl_c = np.ascontiguousarray(dcl[:lmaxCMB_+1], dtype=np.float64)
	elif nspec==4:
		wcl_c = [np.ascontiguousarray(cl[:lmaxCMB_+1], dtype=np.float64) for cl in wcl]
		dcl_c = [np.ascontiguousarray(cl[:lmaxCMB_+1], dtype=np.float64) for cl in dcl]

	if nspec==1:
		wcl_=ndarray2cl1(wcl_c, lmaxCMB_)
		dcl_=ndarray2cl1(dcl_c, lmaxCMB_)
	elif nspec==4:
		wcl_=ndarray2cl4(wcl_c[0], wcl_c[1], wcl_c[2], wcl_c[3], lmaxCMB_)
		dcl_=ndarray2cl4(dcl_c[0], dcl_c[1], dcl_c[2], dcl_c[3], lmaxCMB_)

	alms_cA = np.ascontiguousarray(alms[t2i(type[0])], dtype=np.complex128)
	if alms2 is None:
		alms_cB = np.ascontiguousarray(alms[t2i(type[1])], dtype=np.complex128)
	else:
		alms_cB = np.ascontiguousarray(alms2[t2i(type[1])], dtype=np.complex128)

	A = ndarray2alm(alms_cA, lmaxCMB_, lmaxCMB_)
	B = ndarray2alm(alms_cB, lmaxCMB_, lmaxCMB_)

	n_alm = alm_getn(lmax_, lmax_)
	almP = np.empty(n_alm, dtype=np.complex128)
	AP = ndarray2alm(almP, lmax_, lmax_)

	est_rot(A[0], B[0], type_, AP[0], wcl_[0], dcl_[0], lmin_, lminCMB1_, lminCMB2_, lmaxCMB1_, lmaxCMB2_, nside_)

	del A, B, AP, wcl_, dcl_
	
	return almP
    
def quest_spinflip(alms, wcl, dcl, type, nside, lmin, lmax, lminCMB, lmaxCMB, lminCMB2=None, lmaxCMB2=None, alms2=None):
	"""Estimate phi from a set of 2 spinned alm
	Parameters
	---------- 
	alms : list of 1, 2 or 3 arrays
	  T only, Q and U or T,Q and U maps in one list
	wcl : array-like, shape (1,lmaxCMB) or (4, lmaxCMB)
	  The input power spectra (lensed or unlensed) used in the weights of the normalization a la Okamoto & Hu, either TT only or TT, EE, BB and TE (polarization).
	dcl : array-like, shape (1,lmaxCMB) or (4, lmaxCMB)
	  The (noisy) input power spectra used in the denominators of the normalization (i.e. Wiener filtering), either TT only or TT, EE, BB and TE (polarization).
	type : string
		TT, TE, EE, TB, EB or BB
	lmin : int, scalar, optional
	  Minimum l of the normalization. Default: 2
	lmax : int, scalar, optional
	  Maximum l of the normalization. Default: lmaxCMB
	lminCMB : int, scalar, optional
	  Minimum l of the CMB power spectra. Default: 2
	lmaxCMB : int, scalar, optional
	  Maximum l of the CMB power spectra. Default: given by input cl arrays
		  
	Returns
	-------
	almP : array
		array containing the estimated lensing potential in harmonic space, unnormalized
	"""
	
	nspec=cltype(wcl)
	cdef int lmin_, lmax_, lminCMB_, lmaxCMB_,lminCMB1_, lmaxCMB1_,lminCMB2_, lmaxCMB2_, nside_
	cdef string type_
	type_=type.encode()
	
	lmin_=lmin
	lminCMB1_=lminCMB
	lminCMB2_=lminCMB2
	
	lmax_=lmax	
	lmaxCMB1_=lmaxCMB
	lmaxCMB2_=lmaxCMB2
	
	lmaxCMB_=np.max([lmaxCMB,lmaxCMB2])
	
	nside_=nside
		
	if nspec==1:
		wcl_c = np.ascontiguousarray(wcl[:lmaxCMB_+1], dtype=np.float64)
		dcl_c = np.ascontiguousarray(dcl[:lmaxCMB_+1], dtype=np.float64)
	elif nspec==4:
		wcl_c = [np.ascontiguousarray(cl[:lmaxCMB_+1], dtype=np.float64) for cl in wcl]
		dcl_c = [np.ascontiguousarray(cl[:lmaxCMB_+1], dtype=np.float64) for cl in dcl]

	if nspec==1:
		wcl_=ndarray2cl1(wcl_c, lmaxCMB_)
		dcl_=ndarray2cl1(dcl_c, lmaxCMB_)
	elif nspec==4:
		wcl_=ndarray2cl4(wcl_c[0], wcl_c[1], wcl_c[2], wcl_c[3], lmaxCMB_)
		dcl_=ndarray2cl4(dcl_c[0], dcl_c[1], dcl_c[2], dcl_c[3], lmaxCMB_)

	alms_cA = np.ascontiguousarray(alms[t2i(type[0])], dtype=np.complex128)
	if alms2 is None:
		alms_cB = np.ascontiguousarray(alms[t2i(type[1])], dtype=np.complex128)
	else:
		alms_cB = np.ascontiguousarray(alms2[t2i(type[1])], dtype=np.complex128)

	A = ndarray2alm(alms_cA, lmaxCMB_, lmaxCMB_)
	B = ndarray2alm(alms_cB, lmaxCMB_, lmaxCMB_)

	n_alm = alm_getn(lmax_, lmax_)
	almP = np.empty(n_alm, dtype=np.complex128)
	almC = np.empty(n_alm, dtype=np.complex128)
	AP = ndarray2alm(almP, lmax_, lmax_)
	AC = ndarray2alm(almC, lmax_, lmax_)

	est_spinflip(A[0], B[0], type_, AP[0], AC[0], wcl_[0], dcl_[0], lmin_, lminCMB1_, lminCMB2_, lmaxCMB1_, lmaxCMB2_, nside_)

	del A, B, AP, AC, wcl_, dcl_
	
	return almP,almC
			
            
def quest_monoleak(alms, wcl, dcl, type, nside, lmin, lmax, lminCMB, lmaxCMB, lminCMB2=None, lmaxCMB2=None, alms2=None):
	"""Estimate phi from a set of 2 spinned alm
	Parameters
	---------- 
	alms : list of 1, 2 or 3 arrays
	  T only, Q and U or T,Q and U maps in one list
	wcl : array-like, shape (1,lmaxCMB) or (4, lmaxCMB)
	  The input power spectra (lensed or unlensed) used in the weights of the normalization a la Okamoto & Hu, either TT only or TT, EE, BB and TE (polarization).
	dcl : array-like, shape (1,lmaxCMB) or (4, lmaxCMB)
	  The (noisy) input power spectra used in the denominators of the normalization (i.e. Wiener filtering), either TT only or TT, EE, BB and TE (polarization).
	type : string
		TT, TE, EE, TB, EB or BB
	lmin : int, scalar, optional
	  Minimum l of the normalization. Default: 2
	lmax : int, scalar, optional
	  Maximum l of the normalization. Default: lmaxCMB
	lminCMB : int, scalar, optional
	  Minimum l of the CMB power spectra. Default: 2
	lmaxCMB : int, scalar, optional
	  Maximum l of the CMB power spectra. Default: given by input cl arrays
		  
	Returns
	-------
	almP : array
		array containing the estimated lensing potential in harmonic space, unnormalized
	"""
	
	nspec=cltype(wcl)
	cdef int lmin_, lmax_, lminCMB_, lmaxCMB_,lminCMB1_, lmaxCMB1_,lminCMB2_, lmaxCMB2_, nside_
	cdef string type_
	type_=type.encode()
	
	lmin_=lmin
	lminCMB1_=lminCMB
	lminCMB2_=lminCMB2
	
	lmax_=lmax	
	lmaxCMB1_=lmaxCMB
	lmaxCMB2_=lmaxCMB2
	
	lmaxCMB_=np.max([lmaxCMB,lmaxCMB2])
	
	nside_=nside
		
	if nspec==1:
		wcl_c = np.ascontiguousarray(wcl[:lmaxCMB_+1], dtype=np.float64)
		dcl_c = np.ascontiguousarray(dcl[:lmaxCMB_+1], dtype=np.float64)
	elif nspec==4:
		wcl_c = [np.ascontiguousarray(cl[:lmaxCMB_+1], dtype=np.float64) for cl in wcl]
		dcl_c = [np.ascontiguousarray(cl[:lmaxCMB_+1], dtype=np.float64) for cl in dcl]

	if nspec==1:
		wcl_=ndarray2cl1(wcl_c, lmaxCMB_)
		dcl_=ndarray2cl1(dcl_c, lmaxCMB_)
	elif nspec==4:
		wcl_=ndarray2cl4(wcl_c[0], wcl_c[1], wcl_c[2], wcl_c[3], lmaxCMB_)
		dcl_=ndarray2cl4(dcl_c[0], dcl_c[1], dcl_c[2], dcl_c[3], lmaxCMB_)

	alms_cA = np.ascontiguousarray(alms[t2i(type[0])], dtype=np.complex128)
	if alms2 is None:
		alms_cB = np.ascontiguousarray(alms[t2i(type[1])], dtype=np.complex128)
	else:
		alms_cB = np.ascontiguousarray(alms2[t2i(type[1])], dtype=np.complex128)

	A = ndarray2alm(alms_cA, lmaxCMB_, lmaxCMB_)
	B = ndarray2alm(alms_cB, lmaxCMB_, lmaxCMB_)

	n_alm = alm_getn(lmax_, lmax_)
	almP = np.empty(n_alm, dtype=np.complex128)
	almC = np.empty(n_alm, dtype=np.complex128)
	AP = ndarray2alm(almP, lmax_, lmax_)
	AC = ndarray2alm(almC, lmax_, lmax_)

	est_monoleak(A[0], B[0], type_, AP[0], AC[0], wcl_[0], dcl_[0], lmin_, lminCMB1_, lminCMB2_, lmaxCMB1_, lmaxCMB2_, nside_)

	del A, B, AP, AC, wcl_, dcl_
	
	return almP,almC	
            
def quest_photondir(alms, wcl, dcl, type, nside, lmin, lmax, lminCMB, lmaxCMB, lminCMB2=None, lmaxCMB2=None, alms2=None):
	"""Estimate phi from a set of 2 spinned alm
	Parameters
	---------- 
	alms : list of 1, 2 or 3 arrays
	  T only, Q and U or T,Q and U maps in one list
	wcl : array-like, shape (1,lmaxCMB) or (4, lmaxCMB)
	  The input power spectra (lensed or unlensed) used in the weights of the normalization a la Okamoto & Hu, either TT only or TT, EE, BB and TE (polarization).
	dcl : array-like, shape (1,lmaxCMB) or (4, lmaxCMB)
	  The (noisy) input power spectra used in the denominators of the normalization (i.e. Wiener filtering), either TT only or TT, EE, BB and TE (polarization).
	type : string
		TT, TE, EE, TB, EB or BB
	lmin : int, scalar, optional
	  Minimum l of the normalization. Default: 2
	lmax : int, scalar, optional
	  Maximum l of the normalization. Default: lmaxCMB
	lminCMB : int, scalar, optional
	  Minimum l of the CMB power spectra. Default: 2
	lmaxCMB : int, scalar, optional
	  Maximum l of the CMB power spectra. Default: given by input cl arrays
		  
	Returns
	-------
	almP : array
		array containing the estimated lensing potential in harmonic space, unnormalized
	"""
	
	nspec=cltype(wcl)
	cdef int lmin_, lmax_, lminCMB_, lmaxCMB_,lminCMB1_, lmaxCMB1_,lminCMB2_, lmaxCMB2_, nside_
	cdef string type_
	type_=type.encode()
	
	lmin_=lmin
	lminCMB1_=lminCMB
	lminCMB2_=lminCMB2
	
	lmax_=lmax	
	lmaxCMB1_=lmaxCMB
	lmaxCMB2_=lmaxCMB2
	
	lmaxCMB_=np.max([lmaxCMB,lmaxCMB2])
	
	nside_=nside
		
	if nspec==1:
		wcl_c = np.ascontiguousarray(wcl[:lmaxCMB_+1], dtype=np.float64)
		dcl_c = np.ascontiguousarray(dcl[:lmaxCMB_+1], dtype=np.float64)
	elif nspec==4:
		wcl_c = [np.ascontiguousarray(cl[:lmaxCMB_+1], dtype=np.float64) for cl in wcl]
		dcl_c = [np.ascontiguousarray(cl[:lmaxCMB_+1], dtype=np.float64) for cl in dcl]

	if nspec==1:
		wcl_=ndarray2cl1(wcl_c, lmaxCMB_)
		dcl_=ndarray2cl1(dcl_c, lmaxCMB_)
	elif nspec==4:
		wcl_=ndarray2cl4(wcl_c[0], wcl_c[1], wcl_c[2], wcl_c[3], lmaxCMB_)
		dcl_=ndarray2cl4(dcl_c[0], dcl_c[1], dcl_c[2], dcl_c[3], lmaxCMB_)

	alms_cA = np.ascontiguousarray(alms[t2i(type[0])], dtype=np.complex128)
	if alms2 is None:
		alms_cB = np.ascontiguousarray(alms[t2i(type[1])], dtype=np.complex128)
	else:
		alms_cB = np.ascontiguousarray(alms2[t2i(type[1])], dtype=np.complex128)

	A = ndarray2alm(alms_cA, lmaxCMB_, lmaxCMB_)
	B = ndarray2alm(alms_cB, lmaxCMB_, lmaxCMB_)

	n_alm = alm_getn(lmax_, lmax_)
	almP = np.empty(n_alm, dtype=np.complex128)
	almC = np.empty(n_alm, dtype=np.complex128)
	AP = ndarray2alm(almP, lmax_, lmax_)
	AC = ndarray2alm(almC, lmax_, lmax_)

	est_phodir(A[0], B[0], type_, AP[0], AC[0], wcl_[0], dcl_[0], lmin_, lminCMB1_, lminCMB2_, lmaxCMB1_, lmaxCMB2_, nside_)

	del A, B, AP, AC, wcl_, dcl_
	
	return almP,almC
			   
def quest_dipleak(alms, wcl, dcl, type, nside, lmin, lmax, lminCMB, lmaxCMB, lminCMB2=None, lmaxCMB2=None, alms2=None):
	"""Estimate phi from a set of 2 spinned alm
	Parameters
	---------- 
	alms : list of 1, 2 or 3 arrays
	  T only, Q and U or T,Q and U maps in one list
	wcl : array-like, shape (1,lmaxCMB) or (4, lmaxCMB)
	  The input power spectra (lensed or unlensed) used in the weights of the normalization a la Okamoto & Hu, either TT only or TT, EE, BB and TE (polarization).
	dcl : array-like, shape (1,lmaxCMB) or (4, lmaxCMB)
	  The (noisy) input power spectra used in the denominators of the normalization (i.e. Wiener filtering), either TT only or TT, EE, BB and TE (polarization).
	type : string
		TT, TE, EE, TB, EB or BB
	lmin : int, scalar, optional
	  Minimum l of the normalization. Default: 2
	lmax : int, scalar, optional
	  Maximum l of the normalization. Default: lmaxCMB
	lminCMB : int, scalar, optional
	  Minimum l of the CMB power spectra. Default: 2
	lmaxCMB : int, scalar, optional
	  Maximum l of the CMB power spectra. Default: given by input cl arrays
		  
	Returns
	-------
	almP : array
		array containing the estimated lensing potential in harmonic space, unnormalized
	"""
	
	nspec=cltype(wcl)
	cdef int lmin_, lmax_, lminCMB_, lmaxCMB_,lminCMB1_, lmaxCMB1_,lminCMB2_, lmaxCMB2_, nside_
	cdef string type_
	type_=type.encode()
	
	lmin_=lmin
	lminCMB1_=lminCMB
	lminCMB2_=lminCMB2
	
	lmax_=lmax	
	lmaxCMB1_=lmaxCMB
	lmaxCMB2_=lmaxCMB2
	
	lmaxCMB_=np.max([lmaxCMB,lmaxCMB2])
	
	nside_=nside
		
	if nspec==1:
		wcl_c = np.ascontiguousarray(wcl[:lmaxCMB_+1], dtype=np.float64)
		dcl_c = np.ascontiguousarray(dcl[:lmaxCMB_+1], dtype=np.float64)
	elif nspec==4:
		wcl_c = [np.ascontiguousarray(cl[:lmaxCMB_+1], dtype=np.float64) for cl in wcl]
		dcl_c = [np.ascontiguousarray(cl[:lmaxCMB_+1], dtype=np.float64) for cl in dcl]

	if nspec==1:
		wcl_=ndarray2cl1(wcl_c, lmaxCMB_)
		dcl_=ndarray2cl1(dcl_c, lmaxCMB_)
	elif nspec==4:
		wcl_=ndarray2cl4(wcl_c[0], wcl_c[1], wcl_c[2], wcl_c[3], lmaxCMB_)
		dcl_=ndarray2cl4(dcl_c[0], dcl_c[1], dcl_c[2], dcl_c[3], lmaxCMB_)

	alms_cA = np.ascontiguousarray(alms[t2i(type[0])], dtype=np.complex128)
	if alms2 is None:
		alms_cB = np.ascontiguousarray(alms[t2i(type[1])], dtype=np.complex128)
	else:
		alms_cB = np.ascontiguousarray(alms2[t2i(type[1])], dtype=np.complex128)

	A = ndarray2alm(alms_cA, lmaxCMB_, lmaxCMB_)
	B = ndarray2alm(alms_cB, lmaxCMB_, lmaxCMB_)

	n_alm = alm_getn(lmax_, lmax_)
	almP = np.empty(n_alm, dtype=np.complex128)
	almC = np.empty(n_alm, dtype=np.complex128)
	AP = ndarray2alm(almP, lmax_, lmax_)
	AC = ndarray2alm(almC, lmax_, lmax_)

	est_dipleak(A[0], B[0], type_, AP[0], AC[0], wcl_[0], dcl_[0], lmin_, lminCMB1_, lminCMB2_, lmaxCMB1_, lmaxCMB2_, nside_)

	del A, B, AP, AC, wcl_, dcl_
	
	return almP,almC
					   
def quest_quadleak(alms, wcl, dcl, type, nside, lmin, lmax, lminCMB, lmaxCMB, lminCMB2=None, lmaxCMB2=None, alms2=None):
	"""Estimate phi from a set of 2 spinned alm
	Parameters
	---------- 
	alms : list of 1, 2 or 3 arrays
	  T only, Q and U or T,Q and U maps in one list
	wcl : array-like, shape (1,lmaxCMB) or (4, lmaxCMB)
	  The input power spectra (lensed or unlensed) used in the weights of the normalization a la Okamoto & Hu, either TT only or TT, EE, BB and TE (polarization).
	dcl : array-like, shape (1,lmaxCMB) or (4, lmaxCMB)
	  The (noisy) input power spectra used in the denominators of the normalization (i.e. Wiener filtering), either TT only or TT, EE, BB and TE (polarization).
	type : string
		TT, TE, EE, TB, EB or BB
	lmin : int, scalar, optional
	  Minimum l of the normalization. Default: 2
	lmax : int, scalar, optional
	  Maximum l of the normalization. Default: lmaxCMB
	lminCMB : int, scalar, optional
	  Minimum l of the CMB power spectra. Default: 2
	lmaxCMB : int, scalar, optional
	  Maximum l of the CMB power spectra. Default: given by input cl arrays
		  
	Returns
	-------
	almP : array
		array containing the estimated lensing potential in harmonic space, unnormalized
	"""
	
	nspec=cltype(wcl)
	cdef int lmin_, lmax_, lminCMB_, lmaxCMB_,lminCMB1_, lmaxCMB1_,lminCMB2_, lmaxCMB2_, nside_
	cdef string type_
	type_=type.encode()
	
	lmin_=lmin
	lminCMB1_=lminCMB
	lminCMB2_=lminCMB2
	
	lmax_=lmax	
	lmaxCMB1_=lmaxCMB
	lmaxCMB2_=lmaxCMB2
	
	lmaxCMB_=np.max([lmaxCMB,lmaxCMB2])
	
	nside_=nside
		
	if nspec==1:
		wcl_c = np.ascontiguousarray(wcl[:lmaxCMB_+1], dtype=np.float64)
		dcl_c = np.ascontiguousarray(dcl[:lmaxCMB_+1], dtype=np.float64)
	elif nspec==4:
		wcl_c = [np.ascontiguousarray(cl[:lmaxCMB_+1], dtype=np.float64) for cl in wcl]
		dcl_c = [np.ascontiguousarray(cl[:lmaxCMB_+1], dtype=np.float64) for cl in dcl]

	if nspec==1:
		wcl_=ndarray2cl1(wcl_c, lmaxCMB_)
		dcl_=ndarray2cl1(dcl_c, lmaxCMB_)
	elif nspec==4:
		wcl_=ndarray2cl4(wcl_c[0], wcl_c[1], wcl_c[2], wcl_c[3], lmaxCMB_)
		dcl_=ndarray2cl4(dcl_c[0], dcl_c[1], dcl_c[2], dcl_c[3], lmaxCMB_)

	alms_cA = np.ascontiguousarray(alms[t2i(type[0])], dtype=np.complex128)
	if alms2 is None:
		alms_cB = np.ascontiguousarray(alms[t2i(type[1])], dtype=np.complex128)
	else:
		alms_cB = np.ascontiguousarray(alms2[t2i(type[1])], dtype=np.complex128)

	A = ndarray2alm(alms_cA, lmaxCMB_, lmaxCMB_)
	B = ndarray2alm(alms_cB, lmaxCMB_, lmaxCMB_)

	n_alm = alm_getn(lmax_, lmax_)
	almP = np.empty(n_alm, dtype=np.complex128)
	almC = np.empty(n_alm, dtype=np.complex128)
	AP = ndarray2alm(almP, lmax_, lmax_)
	AC = ndarray2alm(almC, lmax_, lmax_)

	est_quadleak(A[0], B[0], type_, AP[0], AC[0], wcl_[0], dcl_[0], lmin_, lminCMB1_, lminCMB2_, lmaxCMB1_, lmaxCMB2_, nside_)

	del A, B, AP, AC, wcl_, dcl_
	
	return almP
			
		
def quest_curl(alms, wcl, dcl, type, nside, lmin, lmax, lminCMB, lmaxCMB):
	"""Estimate omega from a set of 2 spinned alm
	Parameters
	---------- 
	alms : list of 1, 2 or 3 arrays
	  T only, Q and U or T,Q and U maps in one list
	wcl : array-like, shape (1,lmaxCMB) or (4, lmaxCMB)
	  The input power spectra (lensed or unlensed) used in the weights of the normalization a la Okamoto & Hu, either TT only or TT, EE, BB and TE (polarization).
	dcl : array-like, shape (1,lmaxCMB) or (4, lmaxCMB)
	  The (noisy) input power spectra used in the denominators of the normalization (i.e. Wiener filtering), either TT only or TT, EE, BB and TE (polarization).
	type : string
		TT, TE, EE, TB, EB or BB
	lmin : int, scalar, optional
	  Minimum l of the normalization. Default: 2
	lmax : int, scalar, optional
	  Maximum l of the normalization. Default: lmaxCMB
	lminCMB : int, scalar, optional
	  Minimum l of the CMB power spectra. Default: 2
	lmaxCMB : int, scalar, optional
	  Maximum l of the CMB power spectra. Default: given by input cl arrays
		  
	Returns
	-------
	almP : array
		array containing the estimated lensing potential in harmonic space, unnormalized
	"""
	
	print('To be included in next version')
	
	return 0
	
	
def quest_mask(alms, wcl, dcl, type, nside, lmin, lmax, lminCMB, lmaxCMB, lminCMB2=None, lmaxCMB2=None, alms2=None):
	"""Estimate mask from a set of 2 spinned alm
	Parameters
	---------- 
	alms : list of 1, 2 or 3 arrays
	  T only, Q and U or T,Q and U maps in one list
	wcl : array-like, shape (1,lmaxCMB) or (4, lmaxCMB)
	  The input power spectra (lensed or unlensed) used in the weights of the normalization a la Okamoto & Hu, either TT only or TT, EE, BB and TE (polarization).
	dcl : array-like, shape (1,lmaxCMB) or (4, lmaxCMB)
	  The (noisy) input power spectra used in the denominators of the normalization (i.e. Wiener filtering), either TT only or TT, EE, BB and TE (polarization).
	type : string
		TT, TE, EE, TB, EB or BB
	lmin : int, scalar, optional
	  Minimum l of the normalization. Default: 2
	lmax : int, scalar, optional
	  Maximum l of the normalization. Default: lmaxCMB
	lminCMB : int, scalar, optional
	  Minimum l of the CMB power spectra. Default: 2
	lmaxCMB : int, scalar, optional
	  Maximum l of the CMB power spectra. Default: given by input cl arrays
		  
	Returns
	-------
	almP : array
		array containing the estimated lensing potential in harmonic space, unnormalized
	"""
	
	nspec=cltype(wcl)
	cdef int lmin_, lmax_, lminCMB_, lmaxCMB_,lminCMB1_, lmaxCMB1_,lminCMB2_, lmaxCMB2_, nside_
	cdef string type_
	type_=type.encode()
	
	lmin_=lmin
	lminCMB_=lminCMB
	lminCMB2_=lminCMB2
	
	lmax_=lmax
	lmaxCMB_=lmaxCMB
	lmaxCMB2_=lmaxCMB2
	
	lmaxCMB_=np.max([lmaxCMB,lmaxCMB2])

	nside_=nside

	if nspec==1:
		wcl_c = np.ascontiguousarray(wcl[:lmaxCMB_+1], dtype=np.float64)
		dcl_c = np.ascontiguousarray(dcl[:lmaxCMB_+1], dtype=np.float64)
	elif nspec==4:
		wcl_c = [np.ascontiguousarray(cl[:lmaxCMB_+1], dtype=np.float64) for cl in wcl]
		dcl_c = [np.ascontiguousarray(cl[:lmaxCMB_+1], dtype=np.float64) for cl in dcl]

	if nspec==1:
		wcl_=ndarray2cl1(wcl_c, lmaxCMB_)
		dcl_=ndarray2cl1(dcl_c, lmaxCMB_)
	elif nspec==4:
		wcl_=ndarray2cl4(wcl_c[0], wcl_c[1], wcl_c[2], wcl_c[3], lmaxCMB_)
		dcl_=ndarray2cl4(dcl_c[0], dcl_c[1], dcl_c[2], dcl_c[3], lmaxCMB_)

	alms_cA = np.ascontiguousarray(alms[t2i(type[0])], dtype=np.complex128)
	if alms2 is None:
		alms_cB = np.ascontiguousarray(alms[t2i(type[1])], dtype=np.complex128)
	else:
		alms_cB = np.ascontiguousarray(alms2[t2i(type[1])], dtype=np.complex128)

	A = ndarray2alm(alms_cA, lmaxCMB_, lmaxCMB_)
	B = ndarray2alm(alms_cB, lmaxCMB_, lmaxCMB_)

	n_alm = alm_getn(lmax_, lmax_)
	almP = np.empty(n_alm, dtype=np.complex128)
	AP = ndarray2alm(almP, lmax_, lmax_)

	est_mask(A[0], B[0], type_, AP[0], wcl_[0], dcl_[0], lmin_, lminCMB1_, lminCMB2_, lmaxCMB1_, lmaxCMB2_, nside_)

	del A, B, AP, wcl_, dcl_
	
	return almP
	
	
def quest_noise(alms, wcl, dcl, type, nside, lmin, lmax, lminCMB, lmaxCMB):
	"""Estimate noise/point sources from a set of 2 spinned alm
	Parameters
	---------- 
	alms : list of 1, 2 or 3 arrays
	  T only, Q and U or T,Q and U maps in one list
	wcl : array-like, shape (1,lmaxCMB) or (4, lmaxCMB)
	  The input power spectra (lensed or unlensed) used in the weights of the normalization a la Okamoto & Hu, either TT only or TT, EE, BB and TE (polarization).
	dcl : array-like, shape (1,lmaxCMB) or (4, lmaxCMB)
	  The (noisy) input power spectra used in the denominators of the normalization (i.e. Wiener filtering), either TT only or TT, EE, BB and TE (polarization).
	type : string
		TT, TE, EE, TB, EB or BB
	lmin : int, scalar, optional
	  Minimum l of the normalization. Default: 2
	lmax : int, scalar, optional
	  Maximum l of the normalization. Default: lmaxCMB
	lminCMB : int, scalar, optional
	  Minimum l of the CMB power spectra. Default: 2
	lmaxCMB : int, scalar, optional
	  Maximum l of the CMB power spectra. Default: given by input cl arrays
		  
	Returns
	-------
	almP : array
		array containing the estimated lensing potential in harmonic space, unnormalized
	"""
	
	nspec=cltype(wcl)
	cdef int lmin_, lmax_, lminCMB_, lmaxCMB_, nside_
	cdef string type_
	type_=type.encode()
	
	lmin_=lmin
	lminCMB_=lminCMB
	lmax_=lmax
	lmaxCMB_=lmaxCMB
	nside_=nside

	if nspec==1:
		wcl_c = np.ascontiguousarray(wcl[:lmaxCMB_+1], dtype=np.float64)
		dcl_c = np.ascontiguousarray(dcl[:lmaxCMB_+1], dtype=np.float64)
	elif nspec==4:
		wcl_c = [np.ascontiguousarray(cl[:lmaxCMB_+1], dtype=np.float64) for cl in wcl]
		dcl_c = [np.ascontiguousarray(cl[:lmaxCMB_+1], dtype=np.float64) for cl in dcl]

	if nspec==1:
		wcl_=ndarray2cl1(wcl_c, lmaxCMB_)
		dcl_=ndarray2cl1(dcl_c, lmaxCMB_)
	elif nspec==4:
		wcl_=ndarray2cl4(wcl_c[0], wcl_c[1], wcl_c[2], wcl_c[3], lmaxCMB_)
		dcl_=ndarray2cl4(dcl_c[0], dcl_c[1], dcl_c[2], dcl_c[3], lmaxCMB_)

	alms_cA = np.ascontiguousarray(alms[t2i(type[0])], dtype=np.complex128)
	alms_cB = np.ascontiguousarray(alms[t2i(type[1])], dtype=np.complex128)

	A = ndarray2alm(alms_cA, lmaxCMB_, lmaxCMB_)
	B = ndarray2alm(alms_cB, lmaxCMB_, lmaxCMB_)

	n_alm = alm_getn(lmax_, lmax_)
	almP = np.empty(n_alm, dtype=np.complex128)
	AP = ndarray2alm(almP, lmax_, lmax_)

	est_noise(A[0], B[0], type_, AP[0], wcl_[0], dcl_[0], lmin_, lminCMB_, nside_)

	del A, B, AP, wcl_, dcl_
	
	return almP
	
	
def Btemplate(almE, almP, nside=None, lminB=2, lmaxB=None, lminE=2, lmaxE=None, lminP=2, lmaxP=None):
	cdef int lminB_, lmaxB_, lminE_, lmaxE_, lminP_, lmaxP_, nside_

	lminB_=lminB
	lminE_=lminE
	lminP_=lminP
	
	if lmaxE!=None:
		lmmapE=almpy.getidx(almpy.getlmax(len(almE)),*almpy.getlm(lmaxE,np.arange(almpy.getsize(lmaxE))))
		lmaxE_=lmaxE
		almE_ = np.ascontiguousarray(almE[lmmapE], dtype=np.complex128)
	else:
		lmaxE_=almpy.getlmax(len(almE))
		almE_ = np.ascontiguousarray(almE, dtype=np.complex128)
		
	if lmaxP!=None:
		lmmapP=almpy.getidx(almpy.getlmax(len(almP)),*almpy.getlm(lmaxP,np.arange(almpy.getsize(lmaxP))))
		lmaxP_=lmaxP
		almP_ = np.ascontiguousarray(almP[lmmapP], dtype=np.complex128)
	else:
		lmaxP_=almpy.getlmax(len(almP))
		almP_ = np.ascontiguousarray(almP, dtype=np.complex128)
		
	if lmaxB!=None:
		lmaxB_=lmaxB
	else:
		lmaxB_=lmaxE_
		
	if nside is None: nside_=get_min_valid_nside(12*(.5*np.max([lmaxE_,lmaxP_]))**2)
	else: nside_=nside

	E = ndarray2alm(almE_, lmaxE_, lmaxE_)
	P = ndarray2alm(almP_, lmaxP_, lmaxP_)

	n_alm = alm_getn(lmaxB_, lmaxB_)
	almB = np.empty(n_alm, dtype=np.complex128)
	B = ndarray2alm(almB, lmaxB_, lmaxB_)

	btemp(B[0], E[0], P[0], lminB_, lminE_, lminP_, nside_)

	del B, E, P
	
	return almB

	
cdef int alm_getn(int l, int m):
	if not m <= l:
		raise ValueError("mmax must be <= lmax")
	return ((m+1)*(m+2))/2 + (m+1)*(l-m)
	
	
def lqmap2alm(maps, lmax = None, mmax = None, niter = 3, spin = 2):
	"""Computes the spinned alm of a 2 Healpix maps.
	Parameters
	----------
	m : list of 3 arrays
	mask : array containing apodized mask 
	lmax : int, scalar, optional
	  Maximum l of the power spectrum. Default: 3*nside-1
	mmax : int, scalar, optional
	  Maximum m of the alm. Default: lmax
	
	Returns
	-------
	pure alms : list of 3 arrays
	"""
	
	maps_c = [np.ascontiguousarray(m, dtype=np.float64) for m in maps]


	# Adjust lmax and mmax
	cdef int lmax_, mmax_, nside, npix
		
	npix = maps_c[0].size
	nside = npix2nside(npix)
	if lmax is None:
		lmax_ = 3 * nside - 1
	else:
		lmax_ = lmax
	if mmax is None:
		mmax_ = lmax_
	else:
		mmax_ = mmax

	# Check all maps have same npix
	if maps_c[1].size != npix:
		raise ValueError("Input maps must have same size")	  
	
	# View the ndarray as a Healpix_Map
	M1 = ndarray2map(maps_c[0], RING)
	M2 = ndarray2map(maps_c[1], RING)

	# Create an ndarray object that will contain the alm for output (to be returned)
	n_alm = alm_getn(lmax_, mmax_)
	alms = [np.empty(n_alm, dtype=np.complex128) for m in maps]

	# View the ndarray as an Alm
	# Alms = [ndarray2alm(alm, lmax_, mmax_) for alm in alms]
	A1 = ndarray2alm(alms[0], lmax_, mmax_) 
	A2 = ndarray2alm(alms[1], lmax_, mmax_) 
	
	map2alm_spin_iter(M1[0], M2[0], A1[0], A2[0], spin, niter)

	del M1, M2, A1, A2
	return alms
    
def lqalm2map(alms, nside, spin = 2):
	"""Computes the spinned alm of a 2 Healpix maps.
	Parameters
	----------
	m : list of 3 arrays
	mask : array containing apodized mask 
	lmax : int, scalar, optional
	  Maximum l of the power spectrum. Default: 3*nside-1
	mmax : int, scalar, optional
	  Maximum m of the alm. Default: lmax
	
	Returns
	-------
	pure alms : list of 3 arrays
	"""
	
	alms_cA = np.ascontiguousarray(alms[0], dtype=np.complex128)
	alms_cB = np.ascontiguousarray(alms[1], dtype=np.complex128)

	# Adjust lmax and mmax	
	lmax_ = almpy.getlmax(alms_cA.size)
	mmax_ = lmax_

	# Check all maps have same npix
	if alms_cB.size != alms_cA.size:
		raise ValueError("Input alms must have same size")	  
	
	# View the ndarray as a Healpix_Map
	maps = [np.empty(nside2npix(nside), dtype=np.float64) for m in alms]
    
	M1 = ndarray2map(maps[0], RING)
	M2 = ndarray2map(maps[1], RING)

	# View the ndarray as an Alm
	# Alms = [ndarray2alm(alm, lmax_, mmax_) for alm in alms]
	A1 = ndarray2alm(alms_cA, lmax_, mmax_) 
	A2 = ndarray2alm(alms_cB, lmax_, mmax_) 
	
	alm2map_spin(A1[0], A2[0], M1[0], M2[0], spin)

	del M1, M2, A1, A2
	return maps
