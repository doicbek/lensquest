# coding: utf-8

import numpy as np
cimport numpy as np
from libcpp.string cimport string
from libc.math cimport sqrt, floor, fabs
from libcpp.vector cimport vector
cimport libc
import os
import cython
from libcpp cimport bool as cbool

from _common cimport tsize, arr, PowSpec, ndarray2cl1, ndarray2cl4, ndarray2cl6
								
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
							   
cdef extern from "_lensquest_cxx.cpp":
	cdef void makeA(PowSpec &wcl, PowSpec &dcl, PowSpec &al, int lmin, int lmax, int lminCMB)
	cdef void makeA_syst(PowSpec &wcl, PowSpec &dcl, PowSpec &al, int lmin, int lmax, int lminCMB, int type)
	cdef vector[ vector[double] ] makeAN(PowSpec &wcl, PowSpec &dcl, PowSpec &rdcls, PowSpec &al, int lmin, int lmax, int lminCMB1, int lminCMB2, int lmaxCMB1, int lmaxCMB2)
	cdef vector[ vector[double] ] makeA_BH(string stype, PowSpec &wcl, PowSpec &dcl, int lmin, int lmax, int lminCMB)
	cdef vector[ vector[double] ] computeKernel(string stype, PowSpec &wcl, PowSpec &dcl, int lminCMB, int L)
	cdef void lensCls(PowSpec& llcl, PowSpec& ulcl,vector[double] &clDD) 
	cdef vector[double] lensBB(vector[double] &clEE, vector[double] &clDD, int lmax_out, cbool even)

def quest_norm(wcl, dcl, lmin=2, lmax=None, lminCMB=2, lmaxCMB=None, lminCMB2=None, lmaxCMB2=None, rdcl=None, bias=False):
	"""Computes the norm of the quadratic estimator.

	Parameters
	----------
	wcl : array-like, shape (1,lmaxCMB) or (4, lmaxCMB)
	  The input power spectra (lensed or unlensed) used in the weights of the normalization a la Okamoto & Hu, either TT only or TT, EE, BB and TE (polarization).
	dcl : array-like, shape (1,lmaxCMB) or (4, lmaxCMB)
	  The (noisy) input power spectra used in the denominators of the normalization (i.e. Wiener filtering), either TT only or TT, EE, BB and TE (polarization).
	lmin : int, scalar, optional
	  Minimum l of the normalization. Default: 2
	lmax : int, scalar, optional
	  Maximum l of the normalization. Default: lmaxCMB
	lminCMB : int, scalar, optional
	  Minimum l of the CMB power spectra. Default: 2
	lmaxCMB : int, scalar, optional
	  Maximum l of the CMB power spectra. Default: given by input cl arrays
	bias: bool, scalar, optional
	  Additionally computing the N0 bias. Default: False
	
	Returns
	-------
	AL: array or tuple of arrays
	  Normalization for 1 (TT) or 5 (TT,TE,EE,TB,EB) quadratic estimators.
	"""
	
	cdef int nspec, nspecout

	nspec=cltype(wcl)
	
	if nspec!=cltype(dcl) or nspec<1: raise ValueError("The two power spectra arrays must be of same type and size")
	if rdcl is not None:
		if nspec!=cltype(rdcl): raise ValueError("The third power spectrum array must be of same type and size")
	if nspec!=1 and nspec!=4: raise NotImplementedError("Power spectra must be given in an array of 1 (TT) or 4 (TT,EE,BB,TE) spectra")
	if bias and nspec!=4: raise NotImplementedError("Need polarization spectra for bias computation")
	 
	cdef int lmin_, lmax_, lminCMB1_, lminCMB2_, lmaxCMB_, lmaxCMB1_, lmaxCMB2_
	lmin_=lmin
	lminCMB1_=lminCMB
	if lmaxCMB is None:
		if nspec==1:
			lmaxCMB=len(wcl)-1
		else:
			lmaxCMB=len(wcl[0])-1

	lmaxCMB_=lmaxCMB
	lmaxCMB1_=lmaxCMB
		
	if lmaxCMB2 is None:
		lmaxCMB2_=lmaxCMB_
	else:
		lmaxCMB2_=lmaxCMB2
		lmaxCMB_=np.max([lmaxCMB,lmaxCMB2])
	
	if lminCMB2 is None:
		lminCMB2_=lminCMB1_
	else:
		lminCMB2_=lminCMB2
		
	
	if lmax is None:
		lmax_=lmaxCMB_
	else:
		lmax_=lmax
		
	if nspec==1:
		wcl_c = np.ascontiguousarray(wcl[:lmaxCMB_+1], dtype=np.float64)
		dcl_c = np.ascontiguousarray(dcl[:lmaxCMB_+1], dtype=np.float64)
		nspec=1
	elif nspec==4:
		wcl_c = [np.ascontiguousarray(cl[:lmaxCMB_+1], dtype=np.float64) for cl in wcl]
		dcl_c = [np.ascontiguousarray(cl[:lmaxCMB_+1], dtype=np.float64) for cl in dcl]
		if rdcl is not None and bias: rdcl_c = [np.ascontiguousarray(cl[:lmaxCMB_+1], dtype=np.float64) for cl in rdcl]
		elif bias: rdcl_c=dcl_c
		nspec=4
	
	nspecout=0

	if nspec==1:
		wcl_=ndarray2cl1(wcl_c, lmaxCMB_)
		dcl_=ndarray2cl1(dcl_c, lmaxCMB_)
		nspecout=1
	elif nspec==4:
		wcl_=ndarray2cl4(wcl_c[0], wcl_c[1], wcl_c[2], wcl_c[3], lmaxCMB_)
		dcl_=ndarray2cl4(dcl_c[0], dcl_c[1], dcl_c[2], dcl_c[3], lmaxCMB_)
		if bias: rdcl_=ndarray2cl4(rdcl_c[0], rdcl_c[1], rdcl_c[2], rdcl_c[3], lmaxCMB_)
		
		nspecout=6

	cdef PowSpec *al_=new PowSpec(nspecout,lmax_)
	cdef vector[ vector[double] ] bias_

	if bias:
		bias_=makeAN(wcl_[0], dcl_[0], rdcl_[0], al_[0], lmin_, lmax_, lminCMB1_, lminCMB2_,lmaxCMB1_, lmaxCMB2_)
		del rdcl_
	else:
		makeA(wcl_[0], dcl_[0], al_[0], lmin_, lmax_, lminCMB1_)
	
	del wcl_, dcl_
		
	if nspecout==1:
		al={}
		if bias: nl={}
		al["TT"]= np.zeros(lmax_+1, dtype=np.float64)
		for l in xrange(lmin,lmax_+1):
			al["TT"][l]=al_.tt(l)
		if bias: nl["TTTT"]=al["TT"]
	elif nspecout==6:
		al={}
		if bias: nl={}
		al["TT"] = np.zeros(lmax_+1, dtype=np.float64)
		al["TE"] = np.zeros(lmax_+1, dtype=np.float64)
		al["EE"] = np.zeros(lmax_+1, dtype=np.float64)
		al["TB"] = np.zeros(lmax_+1, dtype=np.float64)
		al["EB"] = np.zeros(lmax_+1, dtype=np.float64)
		if bias: 
			nl["TTTT"]=np.zeros(lmax_+1, dtype=np.float64)
			nl["TTTE"]=np.zeros(lmax_+1, dtype=np.float64)
			nl["TTEE"]=np.zeros(lmax_+1, dtype=np.float64)
			nl["TETE"]=np.zeros(lmax_+1, dtype=np.float64)
			nl["TEEE"]=np.zeros(lmax_+1, dtype=np.float64)
			nl["EEEE"]=np.zeros(lmax_+1, dtype=np.float64)
			nl["TBTB"]=np.zeros(lmax_+1, dtype=np.float64)
			nl["TBEB"]=np.zeros(lmax_+1, dtype=np.float64)
			nl["EBEB"]=np.zeros(lmax_+1, dtype=np.float64)
		for l in xrange(lmin,lmax_+1):
			al["TT"][l]=al_.tt(l)
			al["TE"][l]=al_.tg(l)
			al["EE"][l]=al_.gg(l)
			al["TB"][l]=al_.tc(l)
			al["EB"][l]=al_.gc(l)
			if bias: 
				nl["TTTT"][l]=bias_[0][l]
				nl["TTTE"][l]=bias_[1][l]
				nl["TTEE"][l]=bias_[2][l]
				nl["TETE"][l]=bias_[3][l]
				nl["TEEE"][l]=bias_[4][l]
				nl["EEEE"][l]=bias_[5][l]
				nl["TBTB"][l]=bias_[6][l]
				nl["TBEB"][l]=bias_[7][l]
				nl["EBEB"][l]=bias_[8][l]
				
	del al_
		
	if bias: 
		return al,nl
	else: 
		return al
		

def quest_norm_syst(type, wcl, dcl, lmin=2, lmax=None, lminCMB=2, lmaxCMB=None, lminCMB2=None, lmaxCMB2=None, rdcl=None):
	"""Computes the norm of the quadratic estimator.

	Parameters
	----------
	wcl : array-like, shape (1,lmaxCMB) or (4, lmaxCMB)
	  The input power spectra (lensed or unlensed) used in the weights of the normalization a la Okamoto & Hu, either TT only or TT, EE, BB and TE (polarization).
	dcl : array-like, shape (1,lmaxCMB) or (4, lmaxCMB)
	  The (noisy) input power spectra used in the denominators of the normalization (i.e. Wiener filtering), either TT only or TT, EE, BB and TE (polarization).
	lmin : int, scalar, optional
	  Minimum l of the normalization. Default: 2
	lmax : int, scalar, optional
	  Maximum l of the normalization. Default: lmaxCMB
	lminCMB : int, scalar, optional
	  Minimum l of the CMB power spectra. Default: 2
	lmaxCMB : int, scalar, optional
	  Maximum l of the CMB power spectra. Default: given by input cl arrays
	bias: bool, scalar, optional
	  Additionally computing the N0 bias. Default: False
	
	Returns
	-------
	AL: array or tuple of arrays
	  Normalization for 1 (TT) or 5 (TT,TE,EE,TB,EB) quadratic estimators.
	"""
	
	cdef int nspec, nspecout

	nspec=cltype(wcl)
	
	if nspec!=cltype(dcl) or nspec<1: raise ValueError("The two power spectra arrays must be of same type and size")
	if rdcl is not None:
		if nspec!=cltype(rdcl): raise ValueError("The third power spectrum array must be of same type and size")
	if nspec!=1 and nspec!=4: raise NotImplementedError("Power spectra must be given in an array of 1 (TT) or 4 (TT,EE,BB,TE) spectra")
	 
	cdef int lmin_, lmax_, lminCMB1_, lminCMB2_, lmaxCMB_, lmaxCMB1_, lmaxCMB2_, type_
	lmin_=lmin
	lminCMB1_=lminCMB
	if lmaxCMB is None:
		if nspec==1:
			lmaxCMB=len(wcl)-1
		else:
			lmaxCMB=len(wcl[0])-1

	lmaxCMB_=lmaxCMB
	lmaxCMB1_=lmaxCMB
		
	if lmaxCMB2 is None:
		lmaxCMB2_=lmaxCMB_
	else:
		lmaxCMB2_=lmaxCMB2
		lmaxCMB_=np.max([lmaxCMB,lmaxCMB2])
	
	if lminCMB2 is None:
		lminCMB2_=lminCMB1_
	else:
		lminCMB2_=lminCMB2
		
	
	if lmax is None:
		lmax_=lmaxCMB_
	else:
		lmax_=lmax
        
	type_=type
		

	wcl_c = [np.ascontiguousarray(cl[:lmaxCMB_+1], dtype=np.float64) for cl in wcl]
	dcl_c = [np.ascontiguousarray(cl[:lmaxCMB_+1], dtype=np.float64) for cl in dcl]
	if rdcl is not None: rdcl_c = [np.ascontiguousarray(cl[:lmaxCMB_+1], dtype=np.float64) for cl in rdcl]
	nspec=4
	

	wcl_=ndarray2cl4(wcl_c[0], wcl_c[1], wcl_c[2], wcl_c[3], lmaxCMB_)
	dcl_=ndarray2cl4(dcl_c[0], dcl_c[1], dcl_c[2], dcl_c[3], lmaxCMB_)
	
	nspecout=6

	cdef PowSpec *al_=new PowSpec(nspecout,lmax_)

	makeA_syst(wcl_[0], dcl_[0], al_[0], lmin_, lmax_, lminCMB1_, type_)
	
	del wcl_, dcl_
		
	al={}
	al["TE"] = np.zeros(lmax_+1, dtype=np.float64)
	al["EE"] = np.zeros(lmax_+1, dtype=np.float64)
	al["TB"] = np.zeros(lmax_+1, dtype=np.float64)
	al["EB"] = np.zeros(lmax_+1, dtype=np.float64)
	for l in xrange(lmin,lmax_+1):
		al["TE"][l]=al_.tg(l)
		al["EE"][l]=al_.gg(l)
		al["TB"][l]=al_.tc(l)
		al["EB"][l]=al_.gc(l)

				
	del al_
		
	return al
		

class quest_kernel:
	def __init__(self, type, wcl, dcl, lminCMB=2, lmaxCMB=None):
		self.type=type
		self.wcl=wcl
		self.dcl=dcl
		self.lminCMB=lminCMB
		self.lmaxCMB=lmaxCMB
		
	def get_kernel(self,L):
		return computekernel(L,self.type,self.wcl,self.dcl,lminCMB=self.lminCMB,lmaxCMB=self.lmaxCMB)
	
	
def computekernel(L,type,wcl,dcl,lminCMB=2,lmaxCMB=None):
	cdef int lminCMB_, L_
	nspec=cltype(wcl)
	
	cdef string type_
	type_=type.encode()
	
	if nspec!=cltype(dcl) or nspec<1: raise ValueError("The two power spectra arrays must be of same type and size")
	if nspec!=1 and nspec!=4: raise NotImplementedError("Power spectra must be given in an array of 1 (TT) or 4 (TT,EE,BB,TE) spectra")
	
	lminCMB_=lminCMB
	if lmaxCMB is None:
		if nspec==1:
			lmaxCMB=len(wcl)-1
		else:
			lmaxCMB=len(wcl[0])-1
	
	if nspec==1:
		wcl_c = np.ascontiguousarray(wcl[:lmaxCMB+1], dtype=np.float64)
		dcl_c = np.ascontiguousarray(dcl[:lmaxCMB+1], dtype=np.float64)
		wcl_=ndarray2cl1(wcl_c, lmaxCMB)
		dcl_=ndarray2cl1(dcl_c, lmaxCMB)
		nspec=1
	elif nspec==4:
		wcl_c = [np.ascontiguousarray(cl[:lmaxCMB+1], dtype=np.float64) for cl in wcl]
		dcl_c = [np.ascontiguousarray(cl[:lmaxCMB+1], dtype=np.float64) for cl in dcl]
		wcl_=ndarray2cl4(wcl_c[0], wcl_c[1], wcl_c[2], wcl_c[3], lmaxCMB)
		dcl_=ndarray2cl4(dcl_c[0], dcl_c[1], dcl_c[2], dcl_c[3], lmaxCMB)
		nspec=4

	cdef vector[ vector[double] ] kernel_
	L_=L
	
	kernel_=computeKernel(type_,wcl_[0], dcl_[0], lminCMB_, L_)
	del wcl_, dcl_
		
	kernel= np.zeros((lmaxCMB+1,lmaxCMB+1), dtype=np.float64)
	for l1 in xrange(lmaxCMB+1):
		for l2 in xrange(lmaxCMB+1):
			kernel[l1][l2]=kernel_[l1][l2]
				
	return kernel
	

def quest_norm_bh(type, wcl, dcl, lmin=2, lmax=None, lminCMB=2, lmaxCMB=None):
	"""Computes the norm of the quadratic estimator.

	Parameters
	----------
	wcl : array-like, shape (1,lmaxCMB) or (4, lmaxCMB)
	  The input power spectra (lensed or unlensed) used in the weights of the normalization a la Okamoto & Hu, either TT only or TT, EE, BB and TE (polarization).
	dcl : array-like, shape (1,lmaxCMB) or (4, lmaxCMB)
	  The (noisy) input power spectra used in the denominators of the normalization (i.e. Wiener filtering), either TT only or TT, EE, BB and TE (polarization).
	lmin : int, scalar, optional
	  Minimum l of the normalization. Default: 2
	lmax : int, scalar, optional
	  Maximum l of the normalization. Default: lmaxCMB
	lminCMB : int, scalar, optional
	  Minimum l of the CMB power spectra. Default: 2
	lmaxCMB : int, scalar, optional
	  Maximum l of the CMB power spectra. Default: given by input cl arrays
	bias: bool, scalar, optional
	  Additionally computing the N0 bias. Default: False
	
	Returns
	-------
	AL: array or tuple of arrays
	  Normalization for 1 (TT) or 5 (TT,TE,EE,TB,EB) quadratic estimators.
	"""
	
	cdef int nspec, nspecout
	cdef string type_
	type_=type.encode()
	
	nspec=cltype(wcl)
	
	if nspec!=cltype(dcl) or nspec<1: raise ValueError("The two power spectra arrays must be of same type and size")
	if nspec!=1 and nspec!=4: raise NotImplementedError("Power spectra must be given in an array of 1 (TT) or 4 (TT,EE,BB,TE) spectra")
	 
	cdef int lmin_, lmax_, lminCMB_, lmaxCMB_
	lmin_=lmin
	lminCMB_=lminCMB
	if lmaxCMB is None:
		if nspec==1:
			lmaxCMB_=len(wcl)-1
		else:
			lmaxCMB_=len(wcl[0])-1
	else:
		lmaxCMB_=lmaxCMB
	if lmax is None:
		lmax_=lmaxCMB_
	else:
		lmax_=lmax
		
	num_bh_types=3

	if nspec==1:
		wcl_c = np.ascontiguousarray(wcl[:lmaxCMB_+1], dtype=np.float64)
		dcl_c = np.ascontiguousarray(dcl[:lmaxCMB_+1], dtype=np.float64)
		nspec=1
	elif nspec==4:
		wcl_c = [np.ascontiguousarray(cl[:lmaxCMB_+1], dtype=np.float64) for cl in wcl]
		dcl_c = [np.ascontiguousarray(cl[:lmaxCMB_+1], dtype=np.float64) for cl in dcl]
		nspec=4
	
	nspecout=0

	if nspec==1:
		wcl_=ndarray2cl1(wcl_c, lmaxCMB_)
		dcl_=ndarray2cl1(dcl_c, lmaxCMB_)
		nspecout=1
	elif nspec==4:
		wcl_=ndarray2cl4(wcl_c[0], wcl_c[1], wcl_c[2], wcl_c[3], lmaxCMB_)
		dcl_=ndarray2cl4(dcl_c[0], dcl_c[1], dcl_c[2], dcl_c[3], lmaxCMB_)		
		nspecout=6

	cdef vector[ vector[double] ] al_

	al_=makeA_BH(type_, wcl_[0], dcl_[0], lmin_, lmax_, lminCMB_)
	
	del wcl_, dcl_
	
	al={}
	al["gg"] = np.zeros(lmax_+1, dtype=np.float64)
	al["gm"] = np.zeros(lmax_+1, dtype=np.float64)
	al["gn"] = np.zeros(lmax_+1, dtype=np.float64)
	al["mm"] = np.zeros(lmax_+1, dtype=np.float64)
	al["mn"] = np.zeros(lmax_+1, dtype=np.float64)
	al["nn"] = np.zeros(lmax_+1, dtype=np.float64)
	for l in xrange(lmin,lmax_+1):
		al["gg"][l]=al_[0][l]
		al["gm"][l]=al_[1][l]
		al["gn"][l]=al_[2][l]
		al["mm"][l]=al_[3][l]
		al["mn"][l]=al_[4][l]
		al["nn"][l]=al_[5][l]
	
	return al
	
def lensbb(clee,clpp,lmax=None,even=False):
	lmaxee=len(clee)-1
	lmaxpp=len(clpp)-1
	if lmax is None:
		lmax=lmaxee
	
	cdef lmax_
	lmax_=lmax
	cdef even_
	even_=even
	
	cdef vector[double] clee_ = vector[double](lmaxee+1,0.)
	cdef vector[double] clpp_ = vector[double](lmaxpp+1,0.)

	for l in range(lmaxee+1):
		clee_[l]=clee[l]
		
	for l in range(lmaxpp+1):
		clpp_[l]=clpp[l]
		
	cdef vector[double] clbb_
	
	clbb_=lensBB(clee_,clpp_,lmax_,even_)
		
	out=np.zeros(lmax_+1, dtype=np.float64)
	for l in range(lmax_+1):
		out[l]=clbb_[l]
		
	return out
	
def lenscls(ucl,clpp):
	cdef int lmax_
	
	lmax_ul=len(ucl[0])-1
	lmaxpp=len(clpp)-1

	ucl_c = [np.ascontiguousarray(cl[:lmax_ul+1], dtype=np.float64) for cl in ucl]
	if len(ucl_c)==6: ucl_=ndarray2cl6(ucl_c[0], ucl_c[1], ucl_c[2], ucl_c[3], ucl_c[4], ucl_c[5], lmax_ul)
	else: raise NotImplementedError('Input spectra should be in an array of length 6: %d given'%len(ucl_c))
		
	
	cdef vector[double] clpp_ = vector[double](lmaxpp+1,0.)
	for l in range(lmaxpp+1):
		clpp_[l]=clpp[l]
		
	lmax_=lmax_ul
	
	cdef PowSpec *lcl_=new PowSpec(6,lmax_)
	
	lensCls(lcl_[0], ucl_[0], clpp_)
	
	l=np.arange(len(clpp))
	R=.5*np.sum(l*(l+1)*(2*l+1)/4./np.pi*clpp)
	
	out=np.zeros((6,lmax_+1), dtype=np.float64)
	for l in xrange(2,lmax_+1):
		out[0][l]=lcl_.tt(l)-l*(l+1)*R*ucl[0][l]
		out[1][l]=lcl_.gg(l)-(l**2+l-4)*R*ucl[1][l]
		out[2][l]=lcl_.cc(l)-(l**2+l-4)*R*ucl[2][l]
		out[3][l]=lcl_.tg(l)-(l**2+l-2)*R*ucl[3][l]
		out[4][l]=lcl_.gc(l)-(l**2+l-4)*R*ucl[4][l]
		out[5][l]=lcl_.tc(l)-(l**2+l-2)*R*ucl[5][l]

	return out
