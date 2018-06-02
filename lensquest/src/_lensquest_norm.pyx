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
							   
cdef extern from "_lensquest_cxx.h":
	cdef void makeA(PowSpec &wcl, PowSpec &dcl, PowSpec &al, int lmin, int lmax, int lminCMB)
	cdef vector[ vector[double] ] makeAN(PowSpec &wcl, PowSpec &dcl, PowSpec &rdcls, PowSpec &al, int lmin, int lmax, int lminCMB1, int lminCMB2, int lmaxCMB1, int lmaxCMB2)
	cdef vector[ vector[double] ] makeA_BH(string stype, PowSpec &wcl, PowSpec &dcl, int lmin, int lmax, int lminCMB)
	cdef vector[ vector[double] ] computeKernel(string stype, PowSpec &wcl, PowSpec &dcl, int lminCMB, int L)
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
	

def quest_norm_x(wcl, rdcl, dcl1, dcl2=None, lmin=2, lmax=None, lminCMB=2, lmaxCMB=None, lminCMB2=None, lmaxCMB2=None):
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
	
	if nspec!=cltype(dcl1) or nspec<1: raise ValueError("The two power spectra arrays must be of same type and size")
	if rdcl is not None:
		if cltype(rdcl)!=6: raise ValueError("The third power spectrum array must contain all Cls")
	if nspec!=1 and nspec!=4: raise NotImplementedError("Power spectra must be given in an array of 1 (TT) or 4 (TT,EE,BB,TE) spectra")
	 
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
		
	if dcl2 is None:
		dcl2=dcl1[:]
	
	if lmax is None:
		lmax_=lmaxCMB_
	else:
		lmax_=lmax
		
	if nspec==1:
		wcl_c = np.ascontiguousarray(wcl[:lmaxCMB_+1], dtype=np.float64)
		dcl1_c = np.ascontiguousarray(dcl1[:lmaxCMB_+1], dtype=np.float64)
		dcl2_c = np.ascontiguousarray(dcl2[:lmaxCMB_+1], dtype=np.float64)
		nspec=1
	elif nspec==4:
		wcl_c = [np.ascontiguousarray(cl[:lmaxCMB_+1], dtype=np.float64) for cl in wcl]
		dcl1_c = [np.ascontiguousarray(cl[:lmaxCMB_+1], dtype=np.float64) for cl in dcl1]
		dcl2_c = [np.ascontiguousarray(cl[:lmaxCMB_+1], dtype=np.float64) for cl in dcl2]
		rdcl_c = [np.ascontiguousarray(cl[:lmaxCMB_+1], dtype=np.float64) for cl in rdcl]
		nspec=4
	
	nspecout=0

	if nspec==1:
		wcl_=ndarray2cl1(wcl_c, lmaxCMB_)
		dcl1_=ndarray2cl1(dcl1_c, lmaxCMB1_)
		dcl2_=ndarray2cl1(dcl2_c, lmaxCMB2_)
		nspecout=1
	elif nspec==4:
		wcl_=ndarray2cl4(wcl_c[0], wcl_c[1], wcl_c[2], wcl_c[3], lmaxCMB_)
		dcl1_=ndarray2cl4(dcl1_c[0], dcl1_c[1], dcl1_c[2], dcl1_c[3], lmaxCMB1_)
		dcl2_=ndarray2cl4(dcl2_c[0], dcl2_c[1], dcl2_c[2], dcl2_c[3], lmaxCMB2_)
		rdcl_=ndarray2cl6(rdcl_c[0], rdcl_c[1], rdcl_c[2], rdcl_c[3], rdcl_c[4], rdcl_c[5], lmaxCMB_)
		
		nspecout=6

	cdef PowSpec *al1_=new PowSpec(nspecout,lmax_)
	cdef PowSpec *al2_=new PowSpec(nspecout,lmax_)
	cdef vector[ vector[double] ] bias_

	bias_=makeX_null(wcl_[0], dcl1_[0], dcl2_[0], rdcl_[0], al1_[0], al2_[0], lmin_, lmax_, lminCMB1_, lminCMB2_,lmaxCMB1_, lmaxCMB2_)
	del rdcl_		
	del wcl_, dcl1_, dcl2_
		
	if nspecout==1:
		al={}
		nl={}
		al["TT"]= np.zeros((2,lmax_+1), dtype=np.float64)
		for l in xrange(lmin,lmax_+1):
			al["TT"][0][l]=al1_.tt(l)
			al["TT"][1][l]=al2_.tt(l)
			nl["TTTT"]=al["TT"]
	elif nspecout==6:
		al={}
		nl={}
		al["TT"] = np.zeros((2,lmax_+1), dtype=np.float64)
		al["TE"] = np.zeros((2,lmax_+1), dtype=np.float64)
		al["EE"] = np.zeros((2,lmax_+1), dtype=np.float64)
		al["TB"] = np.zeros((2,lmax_+1), dtype=np.float64)
		al["EB"] = np.zeros((2,lmax_+1), dtype=np.float64)

		nl["TTTT"]=np.zeros(lmax_+1, dtype=np.float64)
		nl["TTTE"]=np.zeros(lmax_+1, dtype=np.float64)
		nl["TTEE"]=np.zeros(lmax_+1, dtype=np.float64)
		nl["TTTB"]=np.zeros(lmax_+1, dtype=np.float64)
		nl["TTEB"]=np.zeros(lmax_+1, dtype=np.float64)
		nl["TETT"]=np.zeros(lmax_+1, dtype=np.float64)
		nl["TETE"]=np.zeros(lmax_+1, dtype=np.float64)
		nl["TEEE"]=np.zeros(lmax_+1, dtype=np.float64)
		nl["TETB"]=np.zeros(lmax_+1, dtype=np.float64)
		nl["TEEB"]=np.zeros(lmax_+1, dtype=np.float64)
		nl["EETT"]=np.zeros(lmax_+1, dtype=np.float64)
		nl["EETE"]=np.zeros(lmax_+1, dtype=np.float64)
		nl["EEEE"]=np.zeros(lmax_+1, dtype=np.float64)
		nl["EETB"]=np.zeros(lmax_+1, dtype=np.float64)
		nl["EEEB"]=np.zeros(lmax_+1, dtype=np.float64)
		nl["TBTT"]=np.zeros(lmax_+1, dtype=np.float64)
		nl["TBTE"]=np.zeros(lmax_+1, dtype=np.float64)
		nl["TBEE"]=np.zeros(lmax_+1, dtype=np.float64)
		nl["TBTB"]=np.zeros(lmax_+1, dtype=np.float64)
		nl["TBEB"]=np.zeros(lmax_+1, dtype=np.float64)
		nl["EBTT"]=np.zeros(lmax_+1, dtype=np.float64)
		nl["EBTE"]=np.zeros(lmax_+1, dtype=np.float64)
		nl["EBEE"]=np.zeros(lmax_+1, dtype=np.float64)
		nl["EBTB"]=np.zeros(lmax_+1, dtype=np.float64)
		nl["EBEB"]=np.zeros(lmax_+1, dtype=np.float64)
		for l in xrange(lmin,lmax_+1):
			al["TT"][0][l]=al1_.tt(l)
			al["TE"][0][l]=al1_.tg(l)
			al["EE"][0][l]=al1_.gg(l)
			al["TB"][0][l]=al1_.tc(l)
			al["EB"][0][l]=al1_.gc(l)
			al["TT"][1][l]=al2_.tt(l)
			al["TE"][1][l]=al2_.tg(l)
			al["EE"][1][l]=al2_.gg(l)
			al["TB"][1][l]=al2_.tc(l)
			al["EB"][1][l]=al2_.gc(l)

			nl["TTTT"][l]=bias_[0][l]
			nl["TTTE"][l]=bias_[1][l]
			nl["TTEE"][l]=bias_[2][l]
			nl["TTTB"][l]=bias_[3][l]
			nl["TTEB"][l]=bias_[4][l]
			nl["TETT"][l]=bias_[5][l]
			nl["TETE"][l]=bias_[6][l]
			nl["TEEE"][l]=bias_[7][l]
			nl["TETB"][l]=bias_[8][l]
			nl["TEEB"][l]=bias_[9][l]
			nl["EETT"][l]=bias_[10][l]
			nl["EETB"][l]=bias_[11][l]
			nl["EEEE"][l]=bias_[12][l]
			nl["EETB"][l]=bias_[13][l]
			nl["EEEB"][l]=bias_[14][l]
			nl["TBTT"][l]=bias_[15][l]
			nl["TBTE"][l]=bias_[16][l]
			nl["TBEE"][l]=bias_[17][l]
			nl["TBTB"][l]=bias_[18][l]
			nl["TBEB"][l]=bias_[19][l]
			nl["EBTT"][l]=bias_[20][l]
			nl["EBTE"][l]=bias_[21][l]
			nl["EBEE"][l]=bias_[22][l]
			nl["EBTB"][l]=bias_[23][l]
			nl["EBEB"][l]=bias_[24][l]
			
	del al1_, al2_
		
	return al,nl
	