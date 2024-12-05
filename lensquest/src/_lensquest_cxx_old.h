#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream> 
#include <cassert>
#include <omp.h>

#include <healpix_map.h>
#include <alm.h>
#include <alm_healpix_tools.h>
#include <libsharp/sharp_cxx.h>
#include <powspec.h>
#include <datatypes.h>

#include "wignerSymbols-cpp.cpp"
#include "kernels.cpp"

#include <Python.h>


void alm2map_libsharp(Alm< xcomplex< double > > & almEin, Alm< xcomplex< double > > & almBin, Healpix_Map<double> & mapQout, Healpix_Map<double> & mapUout, size_t spin, arr<double> &weight) {
	int nside = mapQout.Nside();
	size_t lmax = almEin.Lmax();
	sharp_cxxjob<double> job;
	job.set_weighted_Healpix_geometry (nside, &weight[0]);
	job.set_triangular_alm_info (lmax, lmax);
	job.alm2map_spin(&almEin(0,0),&almBin(0,0),&mapQout[0],&mapUout[0],spin,false);
}

void alm2map_libsharp_sY(Alm< xcomplex< double > > & almin, Healpix_Map<double> & mapRout, Healpix_Map<double> & mapIout, int spin, arr<double> &weight) {
	int nside = mapRout.Nside();
	size_t lmax = almin.Lmax();
	Alm< xcomplex< double > > almZ(lmax,lmax);
	sharp_cxxjob<double> job;
	job.set_weighted_Healpix_geometry (nside, &weight[0]);
	job.set_triangular_alm_info (lmax, lmax);
	job.alm2map_spin(&almin(0,0),&almZ(0,0),&mapRout[0],&mapIout[0],abs(spin),false);
}

void map2alm_libsharp(Healpix_Map<double> & mapQin, Healpix_Map<double> & mapUin, Alm< xcomplex< double > > & almEout, Alm< xcomplex< double > > & almBout, size_t spin, arr<double> &weight) {
	int nside = mapQin.Nside();
	int lmax=almEout.Lmax();
	sharp_cxxjob<double> job;
	job.set_weighted_Healpix_geometry (nside, &weight[0]);
	job.set_triangular_alm_info (lmax, lmax);
	job.map2alm_spin(&mapQin[0],&mapUin[0],&almEout(0,0),&almBout(0,0),spin,false);
}

void map2alm_libsharp_sY(Healpix_Map<double> & mapRin, Healpix_Map<double> & mapIin, Alm< xcomplex< double > > & almout, int spin, arr<double> &weight) {
	int nside = mapRin.Nside();
	int lmax=almout.Lmax();
	Alm< xcomplex< double > > almB(lmax,lmax);
	sharp_cxxjob<double> job;
	job.set_weighted_Healpix_geometry (nside, &weight[0]);
	job.set_triangular_alm_info (lmax, lmax);
	job.map2alm_spin(&mapRin[0],&mapIin[0],&almout(0,0),&almB(0,0),abs(spin),false);
	if (spin>=0) {almB.Scale(complex_i);almout.Add(almB);almout.Scale(-1.);}
	else         {almB.Scale(complex_i);almout.Scale(-1.);almout.Add(almB);}
}

void fast_kernel(Alm< xcomplex< double > > & alm1in, Alm< xcomplex< double > > & alm2in, Alm< xcomplex< double > > & almout, size_t spin, int nside, arr<double> &weight) {
	size_t lmaxCMB1=alm1in.Lmax();
	size_t lmaxCMB2=alm1in.Lmax();
	size_t lmax=almout.Lmax();
	
	Healpix_Map<double> map1Q, map1U, map2Q, map2U;
	map1Q.SetNside(nside,RING);
	map1U.SetNside(nside,RING);
	map2Q.SetNside(nside,RING);
	map2U.SetNside(nside,RING);
	
	size_t lmaxtemp=2*nside-1;
	size_t niter=2;
	
	alm2map_libsharp_sY(alm1in,map1Q,map1U,spin,weight);
	alm2map_libsharp_sY(alm2in,map2Q,map2U,spin,weight);
	
	Healpix_Map<double> mapR;
	mapR.SetNside(nside,RING);
	
	#pragma omp parallel for
	for (int i=0; i< map1Q.Npix(); i++) {
		mapR[i]=(map1Q[i]*map2Q[i]+map1U[i]*map2U[i]);
		map1U[i]=(map2Q[i]*map1U[i]-map2U[i]*map1Q[i]);
		if (mapR[i]!=mapR[i]) mapR[i]=0.0;
		if (map1U[i]!=map1U[i]) map1U[i]=0.0;
	}
	
	alm1in.Set(lmaxtemp,lmaxtemp);
	alm2in.Set(lmaxtemp,lmaxtemp);

	map2alm_iter(mapR, alm1in,niter,weight);
	map2alm_iter(map1U,alm2in,niter,weight);

	alm2in.Scale(complex_i);
	alm1in.Add(alm2in);
	
	for (size_t m=0; m<=lmax; ++m) {
		for (size_t l=m; l<=lmax; ++l) {
			almout(l,m)=alm1in(l,m);
		}
	}
}

void fast_kernel_bl(Alm< xcomplex< double > > & alm1in, Alm< xcomplex< double > > & alm2in, Alm< xcomplex< double > > & almout, int nside, arr<double> &weight) {
	size_t lmaxP=alm1in.Lmax();
	size_t lmaxE=alm2in.Lmax();
	size_t lmax=almout.Lmax();
	
	
	Healpix_Map<double> map1Q, map2Q, map2U;
	map1Q.SetNside(nside,RING);
	map2Q.SetNside(nside,RING);
	map2U.SetNside(nside,RING);
	
	size_t lmaxtemp=lmax;//2*nside-1;
	
	alm2map(alm1in,map1Q);
	alm2map_libsharp_sY(alm2in,map2Q,map2U,2,weight);
	
	Healpix_Map<double> mapR, mapI;
	mapR.SetNside(nside,RING);
	
	#pragma omp parallel for
	for (int i=0; i< map1Q.Npix(); i++) {
		mapR[i]=-map1Q[i]*map2Q[i];
		map2Q[i]=-map2U[i]*map1Q[i];
		if (mapR[i]!=mapR[i]) mapR[i]=0.0;
		if (map2Q[i]!=map2Q[i]) map2Q[i]=0.0;
	}
	
	alm1in.Set(lmaxtemp,lmaxtemp);
	alm2in.Set(lmaxtemp,lmaxtemp);

	sharp_cxxjob<double> job;
	job.set_weighted_Healpix_geometry (nside, &weight[0]);
	job.set_triangular_alm_info (lmaxtemp, lmaxtemp);
	job.map2alm_spin(&mapR[0],&map2Q[0],&alm1in(0,0),&alm2in(0,0),2,false);

	alm1in.Scale(-1.*complex_i);
	alm1in.Add(alm2in);
	
	for (size_t m=0; m<=lmax; ++m) {
		for (size_t l=m; l<=lmax; ++l) {
			almout(l,m)=alm1in(l,m);
		}
	}
	
	if(PyErr_CheckSignals() == -1) {
		throw invalid_argument( "Keyboard interrupt" );
	}
}

void compute_term_bl(size_t termnum, Alm< xcomplex< double > > & almB, Alm< xcomplex< double > > & almE, Alm< xcomplex< double > > & almP, size_t lminE, size_t lminP, int nside, arr<double> &weight) {	
	size_t lmaxB=almB.Lmax();
	size_t lmaxE=almE.Lmax();
	size_t lmaxP=almP.Lmax();
	
	Alm< xcomplex< double > > almE_loc(lmaxE,lmaxE), almP_loc(lmaxP,lmaxP);
	
	arr<double> weightE, weightP, lsqr, sgnL;
	weightE.alloc(lmaxE+1);
	weightP.alloc(lmaxP+1);
	weightE.fill(.0);
	weightP.fill(.0);
	size_t spin=0;
	
	lsqr.alloc(max(lmaxE,lmaxP)+1);
	sgnL.alloc(lmaxB+1);
	
	#pragma omp parallel for
	for (size_t l=0; l<lmaxB+1; l++) {
		lsqr[l]=l*(l+1.);
	}
	#pragma omp parallel for
	for (size_t l=0; l<max(lmaxE,lmaxP)+1; l++) {
		sgnL[l]=sgn(l);
	}

	#pragma omp parallel for
	for (size_t l=lminE; l<lmaxE+1; l++) {
		if      (termnum==1) weightE[l]=1.;//wcl.gg(l)/dcl.gg(l);
		else if (termnum==2) weightE[l]=-1.*(l*(l+1.));//wcl.gg(l)/dcl.gg(l);
		else if (termnum==3) weightE[l]=-1.;//wcl.gg(l)/dcl.gg(l);
		else if (termnum==4) weightE[l]=-1.*sgn(l);//wcl.gg(l)/dcl.gg(l);
		else if (termnum==5) weightE[l]=1.*sgn(l)*(l*(l+1.));//wcl.gg(l)/dcl.gg(l);
		else if (termnum==6) weightE[l]=1.*sgn(l);//wcl.gg(l)/dcl.gg(l);
		
		if (weightE[l]!=weightE[l]) weightE[l]=.0;
	}
	
	#pragma omp parallel for
	for (size_t l=lminP; l<lmaxP+1; l++) {
		if      (termnum==1) weightP[l]=1.;//wcl.tt(l)/dcl.tt(l);
		else if (termnum==2) weightP[l]=1.;//wcl.tt(l)/dcl.tt(l);
		else if (termnum==3) weightP[l]=1.*(l*(l+1.));//wcl.tt(l)/dcl.tt(l);
		else if (termnum==4) weightP[l]=1.*sgn(l);//wcl.tt(l)/dcl.tt(l);
		else if (termnum==5) weightP[l]=1.*sgn(l);//wcl.tt(l)/dcl.tt(l);
		else if (termnum==6) weightP[l]=1.*(l*(l+1.))*sgn(l);//wcl.tt(l)/dcl.tt(l);
		
		if (weightP[l]!=weightP[l]) weightP[l]=.0;
	}
	
	#pragma omp parallel for
	for (size_t m=0; m<=lmaxE; ++m) {
		for (size_t l=m; l<=lmaxE; ++l) {
			almE_loc(l,m)=weightE[l]*almE(l,m);
		}
	} 
	#pragma omp parallel for
	for (size_t m=0; m<=lmaxP; ++m) {
		for (size_t l=m; l<=lmaxP; ++l) {
			almP_loc(l,m)=weightP[l]*almP(l,m);
		}
	} 
	
	Alm< xcomplex< double > > almout(lmaxB,lmaxB);
	fast_kernel_bl(almP_loc,almE_loc,almout,nside,weight);

	if (termnum==4 || termnum==5 || termnum==6) {almout.ScaleL(sgnL);}
	if (termnum==1 || termnum==4) almout.ScaleL(lsqr);
	
	almB.Add(almout);
}


void btemp(Alm< xcomplex< double > > &almB, Alm< xcomplex< double > > &almE, Alm< xcomplex< double > > &almP, int lminB, int lminE, int lminP, int nside) {
	size_t lmaxB=almB.Lmax();

	almB.SetToZero();
	
	arr<double> weight;
	weight.alloc(2*nside);
	weight.fill(1.0);
	
	size_t nterms=6;
		
	for (size_t i=1; i<=nterms; i++) {
		compute_term_bl(i, almB, almE, almP, lminE, lminP, nside, weight);
	}
							
	for (size_t m=0; m<=lmaxB; ++m) {
		for (size_t l=m; l<=lmaxB; ++l) {
			almB(l,m)*=.25;
			if (l<lminB) almB(l,m)=0.;
			if (m==0) almB(l,m)=almB(l,m).real();
		}
	}
}



void compute_term_noise(int type, size_t termnum, Alm< xcomplex< double > > & alm1in, Alm< xcomplex< double > > & alm2in, Alm< xcomplex< double > > & almN, PowSpec& wcl, PowSpec& dcl, size_t lmin, size_t lminCMB, int nside, arr<double> &weight) {	
	size_t lmaxCMB=alm1in.Lmax();
	size_t lmax=almN.Lmax();
	
	Alm< xcomplex< double > > alm1(lmaxCMB,lmaxCMB), alm2(lmaxCMB,lmaxCMB);
	
	arr<double> weight1, weight2, lsqr, sgnL;
	weight1.alloc(lmaxCMB+1);
	weight2.alloc(lmaxCMB+1);	
	weight1.fill(.0);
	weight2.fill(.0);
	size_t spin=0;
	
	lsqr.alloc(lmax+1);
	sgnL.alloc(lmax+1);
	
	#pragma omp parallel 
	for (size_t l=0; l<lmax+1; l++) {
		lsqr[l]=l*(l+1.);
		sgnL[l]=sgn(l);
	}
		
	if (type==tt) { 
		#pragma omp parallel 
		for (size_t l=lminCMB; l<lmaxCMB+1; l++) {	
			weight1[l]=1./dcl.tt(l);
			weight2[l]=1./dcl.tt(l);
		}
		spin=0;
	}

	else if (type==te) { 
		#pragma omp parallel 
		for (size_t l=lminCMB; l<lmaxCMB+1; l++) {	
			weight1[l]=1./dcl.tt(l);
			weight2[l]=1./dcl.gg(l);
		}
		spin=2;
	}

	else if (type==ee) { 
		#pragma omp parallel 
		for (size_t l=lminCMB; l<lmaxCMB+1; l++) {	
			weight1[l]=1./dcl.gg(l);
			weight2[l]=1./dcl.gg(l);
		}
		spin=2;
	}

	else if (type==tb) { 
		#pragma omp parallel 
		for (size_t l=lminCMB; l<lmaxCMB+1; l++) {	
			weight1[l]=1./dcl.tt(l);
			weight2[l]=1./dcl.cc(l);
		}
		spin=2;
	}

	else if (type==eb) { 
		#pragma omp parallel 
		for (size_t l=lminCMB; l<lmaxCMB+1; l++) {	
			weight1[l]=1./dcl.gg(l);
			weight2[l]=1./dcl.cc(l);
		}
		spin=2;
	}

	else if (type==bb) { 
		#pragma omp parallel 
		for (size_t l=lminCMB; l<lmaxCMB+1; l++) {	
			weight1[l]=1./dcl.cc(l);
			weight2[l]=1./dcl.cc(l);
		}
		spin=2;
	}

	alm1=alm1in;
	alm1.ScaleL(weight1);
	alm2=alm2in;
	alm2.ScaleL(weight2);
	
	Alm< xcomplex< double > > almout(lmax,lmax);
	fast_kernel(alm1,alm2,almout,spin,nside,weight);
	
	if (type==tt) almout.Scale(.5);
	else if (type==te) almout.Scale(1.);
	else if (type==ee) almout.Scale(.5);
	else if (type==tb) almout.Scale(1.);
	else if (type==eb) almout.Scale(1.);
	else if (type==bb) almout.Scale(.5);
	
	almN.Add(almout);
	
	if(PyErr_CheckSignals() == -1) {
		throw invalid_argument( "Keyboard interrupt" );
	}
}	


void est_noise(Alm< xcomplex< double > > &alm1, Alm< xcomplex< double > > &alm2, std::string stype, Alm< xcomplex< double > > &almN, PowSpec& wcl, PowSpec& dcl, int lmin, int lminCMB, int nside) {
	size_t lmax=almN.Lmax();

	almN.SetToZero();
	int type = string2esttype(stype);
	
	arr<double> weight;
	weight.alloc(2*nside);
	weight.fill(1.0);
	
	size_t nterms=1;
		
	for (size_t i=1; i<=nterms; i++) {
		compute_term_noise(type, i, alm1, alm2, almN, wcl, dcl, lmin, lminCMB, nside, weight);
	}
							
	for (size_t m=0; m<=lmax; ++m) {
		for (size_t l=m; l<=lmax; ++l) {
			if (l<lmin) almN(l,m)=0.;
			if (m==0) almN(l,m)=almN(l,m).real();
		}
	}
}

void compute_term_mask(int type, size_t termnum, Alm< xcomplex< double > > & alm1in, Alm< xcomplex< double > > & alm2in, Alm< xcomplex< double > > & almM, PowSpec& wcl, PowSpec& dcl, size_t lmin, size_t lminCMB1, size_t lminCMB2, size_t lmaxCMB1, size_t lmaxCMB2, int nside, arr<double> &weight) {	
	size_t lmax=almM.Lmax();
	size_t lmaxCMB=max(lmaxCMB1,lmaxCMB2);
	
	Alm< xcomplex< double > > alm1(lmaxCMB,lmaxCMB), alm2(lmaxCMB,lmaxCMB);
	
	arr<double> weight1, weight2, lsqr, sgnL;
	weight1.alloc(lmaxCMB1+1);
	weight2.alloc(lmaxCMB2+1);
	weight1.fill(.0);
	weight2.fill(.0);
	size_t spin=0;
	
	lsqr.alloc(lmax+1);
	sgnL.alloc(lmax+1);
	
	#pragma omp parallel 
	for (size_t l=0; l<lmax+1; l++) {
		lsqr[l]=l*(l+1.);
		sgnL[l]=sgn(l);
	}

	if (type==tt) {
		#pragma omp parallel 
		for (size_t l=lminCMB1; l<lmaxCMB1+1; l++) {
			if 		(termnum==1) weight1[l]=wcl.tt(l)/dcl.tt(l);
			else if (termnum==2) weight1[l]=sgn(l)/dcl.tt(l);
			
			if (weight1[l]!=weight1[l]) weight1[l]=.0;
		}
		
		#pragma omp parallel 
		for (size_t l=lminCMB2; l<lmaxCMB2+1; l++) {
			if 		(termnum==1) weight2[l]=1./dcl.tt(l);
			else if (termnum==2) weight2[l]=wcl.tt(l)*sgn(l)/dcl.tt(l);
			
			if (weight2[l]!=weight2[l]) weight2[l]=.0;
		}
		
		spin=0;
	}
	
	else if (type==te) {
		#pragma omp parallel 
		for (size_t l=lminCMB1; l<lmaxCMB1+1; l++) {
			if		(termnum==1) weight1[l]=wcl.tg(l)/dcl.tt(l);
			else if (termnum==2) weight1[l]=wcl.tg(l)*sgn(l)/dcl.tt(l);
			else if (termnum==3) weight1[l]=1./dcl.tt(l);
			else if (termnum==4) weight1[l]=sgn(l)/dcl.tt(l);
			
			if (weight1[l]!=weight1[l]) weight1[l]=.0;
		}
		
		#pragma omp parallel 
		for (size_t l=lminCMB2; l<lmaxCMB2+1; l++) {
			if 		(termnum==1) weight2[l]=1./dcl.gg(l);
			else if (termnum==2) weight2[l]=sgn(l)/dcl.gg(l);
			else if (termnum==3) weight2[l]=wcl.tg(l)/dcl.gg(l);
			else if (termnum==4) weight2[l]=wcl.tg(l)*sgn(l)/dcl.gg(l);
			
			if (weight2[l]!=weight2[l]) weight2[l]=.0;
		}
		
		if (termnum<=2) spin=2;
		else spin=0;
	}
	
	else if (type==ee) { 
		#pragma omp parallel 
		for (size_t l=lminCMB1; l<lmaxCMB1+1; l++) {
			if  	(termnum==1) weight1[l]=wcl.gg(l)/dcl.gg(l);
			else if (termnum==2) weight1[l]=sgn(l)*wcl.gg(l)/dcl.gg(l);
			else if (termnum==3) weight1[l]=1./dcl.gg(l);
			else if (termnum==4) weight1[l]=sgn(l)/dcl.gg(l);
			
			if (weight1[l]!=weight1[l]) weight1[l]=.0;
		}
		
		#pragma omp parallel 
		for (size_t l=lminCMB2; l<lmaxCMB2+1; l++) {
			if (termnum==1) weight2[l]=1./dcl.gg(l);
			else if (termnum==2) weight2[l]=sgn(l)/dcl.gg(l);
			else if (termnum==3) weight2[l]=wcl.gg(l)/dcl.gg(l);
			else if (termnum==4) weight2[l]=sgn(l)*wcl.gg(l)/dcl.gg(l);
			
			if (weight2[l]!=weight2[l]) weight2[l]=.0;
		}

		spin=2;
	}
	
	else if (type==tb) {
		#pragma omp parallel 
		for (size_t l=lminCMB1; l<lmaxCMB1+1; l++) {
			if 		(termnum==1) weight1[l]=-wcl.tg(l)/dcl.tt(l);
			else if (termnum==2) weight1[l]=sgn(l)*wcl.tg(l)/dcl.tt(l);
			
			if (weight1[l]!=weight1[l]) weight1[l]=.0;
		}
		
		#pragma omp parallel 
		for (size_t l=lminCMB2; l<lmaxCMB2+1; l++) {
			if 		(termnum==1) weight2[l]=1./dcl.cc(l);
			else if (termnum==2) weight2[l]=sgn(l)/dcl.cc(l);
			
			if (weight2[l]!=weight2[l]) weight2[l]=.0;
		}
		
		spin=2;
	}
	
	else if (type==eb) {
		#pragma omp parallel 
		for (size_t l=lminCMB1; l<lmaxCMB1+1; l++) {
			if 		(termnum==1) weight1[l]=-wcl.gg(l)/dcl.gg(l);
			else if (termnum==2) weight1[l]=sgn(l)*wcl.gg(l)/dcl.gg(l);
			
			if (weight1[l]!=weight1[l]) weight1[l]=.0;
		}
		
		#pragma omp parallel 
		for (size_t l=lminCMB2; l<lmaxCMB2+1; l++) {
			if 		(termnum==1) weight2[l]=1./dcl.cc(l);
			else if (termnum==2) weight2[l]=sgn(l)/dcl.cc(l);
			
			if (weight2[l]!=weight2[l]) weight2[l]=.0;
		}
		
		spin=2;
	}
	
	else if (type==bb) { 
		#pragma omp parallel 
		for (size_t l=lminCMB1; l<lmaxCMB1+1; l++) {
			if 		(termnum==1) weight1[l]=wcl.cc(l)/dcl.cc(l);
			else if (termnum==2) weight1[l]=sgn(l)*wcl.cc(l)/dcl.cc(l);
			else if (termnum==3) weight1[l]=1./dcl.cc(l);
			else if (termnum==4) weight1[l]=sgn(l)/dcl.cc(l);

			if (weight1[l]!=weight1[l]) weight1[l]=.0;
		}
		
		#pragma omp parallel 
		for (size_t l=lminCMB2; l<lmaxCMB2+1; l++) {
			if 		(termnum==1) weight2[l]=1./dcl.cc(l);
			else if (termnum==2) weight2[l]=sgn(l)/dcl.cc(l);
			else if (termnum==3) weight2[l]=wcl.cc(l)/dcl.cc(l);
			else if (termnum==4) weight2[l]=sgn(l)*wcl.cc(l)/dcl.cc(l);

			if (weight2[l]!=weight2[l]) weight2[l]=.0;
		}

		spin=2;
	}
	
	#pragma omp parallel for
	for (size_t m=0; m<=lmaxCMB1; ++m) {
		for (size_t l=m; l<=lmaxCMB1; ++l) {
			alm1(l,m)=weight1[l]*alm1in(l,m);
		}
	} 
	#pragma omp parallel for
	for (size_t m=0; m<=lmaxCMB2; ++m) {
		for (size_t l=m; l<=lmaxCMB2; ++l) {
			alm2(l,m)=weight2[l]*alm2in(l,m);
		}
	} 
	
	Alm< xcomplex< double > > almout(lmax,lmax);
	fast_kernel(alm1,alm2,almout,spin,nside,weight);

	if (termnum==2 || termnum==4) almout.ScaleL(sgnL);

	if (type==tt) almout.Scale(1.);
	else if (type==te) almout.Scale(1.);
	else if (type==ee) almout.Scale(.5);
	else if (type==tb) almout.Scale(1.*complex_i);
	else if (type==eb) almout.Scale(1.*complex_i);
	else if (type==bb) almout.Scale(.5);
	
	almM.Add(almout);
	
	if(PyErr_CheckSignals() == -1) {
		throw invalid_argument( "Keyboard interrupt" );
	}
}		

void est_mask(Alm< xcomplex< double > > &alm1, Alm< xcomplex< double > > &alm2, std::string stype, Alm< xcomplex< double > > &almM, PowSpec& wcl, PowSpec& dcl, int lmin, int lminCMB1, int lminCMB2,  int lmaxCMB1, int lmaxCMB2, int nside) {
	size_t lmax=almM.Lmax();

	almM.SetToZero();
	int type = string2esttype(stype);
	
	arr<double> weight;
	weight.alloc(2*nside);
	weight.fill(1.0);
	
	size_t nterms;
	if (type==tt || type==tb || type==eb) nterms=2;
	else nterms=4;		
		
	for (size_t i=1; i<=nterms; i++) {
		compute_term_mask(type, i, alm1, alm2, almM, wcl, dcl, lmin, lminCMB1, lminCMB2, lmaxCMB1, lmaxCMB2, nside, weight);
	}
							
	for (size_t m=0; m<=lmax; ++m) {
		for (size_t l=m; l<=lmax; ++l) {
			almM(l,m)*=.5;
			if (l<lmin) almM(l,m)=0.;
			if (m==0) almM(l,m)=almM(l,m).real();
		}
	}
}

std::vector<double> lensBB(std::vector<double> &clEE, std::vector<double> &clDD, size_t lmax_out, bool even) {
	int lmax_EE=clEE.size()-1;
	int lmax_DD=clDD.size()-1;
	
	std::vector<double> out(lmax_out+1, 0.);
	
	std::vector< std::vector<double> > F(lmax_DD+1, std::vector<double>(lmax_EE+1,0.));
	
	for (size_t l1=0;l1<lmax_out+1;l1++) {
		compF(F, l1, 2, lmax_DD+1, lmax_EE+1);
				
		double Aout=0.;
		if (even==true){
			if (l1>=2) {
				#pragma omp parallel for reduction(+:Aout)
				for (size_t L=2;L<lmax_DD+1;L++) {
					for (size_t l2=2;l2<lmax_EE+1;l2++) {
						if ((l1+L+l2)%2==0) {
							Aout+=F[L][l2]*F[L][l2]*clDD[L]*clEE[l2];
						}
					}
				}
			}
		}
		else {
			if (l1>=2) {
				#pragma omp parallel for reduction(+:Aout)
				for (size_t L=2;L<lmax_DD+1;L++) {
					for (size_t l2=2;l2<lmax_EE+1;l2++) {
						if ((l1+L+l2)%2!=0) {
							Aout+=F[L][l2]*F[L][l2]*clDD[L]*clEE[l2];
						}
					}
				}
			}
		}
		out[l1]=Aout*1./(2.*l1+1.);
						
		if(PyErr_CheckSignals() == -1) {
			throw invalid_argument( "Keyboard interrupt" );
		}
				
	}
	
	return out;
}



void compute_term(int type, size_t termnum, Alm< xcomplex< double > > & alm1in, Alm< xcomplex< double > > & alm2in, Alm< xcomplex< double > > & almP, PowSpec& wcl, PowSpec& dcl, size_t lmin, size_t lminCMB1, size_t lminCMB2, size_t lmaxCMB1, size_t lmaxCMB2, int nside, arr<double> &weight) {	
	size_t lmax=almP.Lmax();
	size_t lmaxCMB=max(lmaxCMB1,lmaxCMB2);
	
	Alm< xcomplex< double > > alm1(lmaxCMB,lmaxCMB), alm2(lmaxCMB,lmaxCMB);
	
	arr<double> weight1, weight2, lsqr, sgnL;
	weight1.alloc(lmaxCMB1+1);
	weight2.alloc(lmaxCMB2+1);
	weight1.fill(.0);
	weight2.fill(.0);
	size_t spin=0;
	
	lsqr.alloc(lmax+1);
	sgnL.alloc(lmax+1);
	
	#pragma omp parallel for
	for (size_t l=0; l<lmax+1; l++) {
		lsqr[l]=l*(l+1.);
		sgnL[l]=sgn(l);
	}

	if (type==tt) {
		#pragma omp parallel for
		for (size_t l=lminCMB1; l<lmaxCMB1+1; l++) {
			if      (termnum==1) weight1[l]=-wcl.tt(l)/dcl.tt(l);
			else if (termnum==2) weight1[l]=wcl.tt(l)/dcl.tt(l);
			else if (termnum==3) weight1[l]=wcl.tt(l)*(l*(l+1.))/dcl.tt(l);
			else if (termnum==4) weight1[l]=-(l*(l+1.))*sgn(l)/dcl.tt(l);
			else if (termnum==5) weight1[l]=1.*sgn(l)/dcl.tt(l);
			else if (termnum==6) weight1[l]=1.*sgn(l)/dcl.tt(l);
			
			if (weight1[l]!=weight1[l]) weight1[l]=.0;
		}
		
		#pragma omp parallel for
		for (size_t l=lminCMB2; l<lmaxCMB2+1; l++) {
			if      (termnum==1) weight2[l]=(l*(l+1.))/dcl.tt(l);
			else if (termnum==2) weight2[l]=1./dcl.tt(l);
			else if (termnum==3) weight2[l]=1./dcl.tt(l);
			else if (termnum==4) weight2[l]=wcl.tt(l)*sgn(l)/dcl.tt(l);
			else if (termnum==5) weight2[l]=wcl.tt(l)*sgn(l)/dcl.tt(l);
			else if (termnum==6) weight2[l]=wcl.tt(l)*(l*(l+1.))*sgn(l)/dcl.tt(l);
			
			if (weight2[l]!=weight2[l]) weight2[l]=.0;
		}
		spin=0;
	}
	
	else if (type==te) { 
		#pragma omp parallel for
		for (size_t l=lminCMB1; l<lmaxCMB1+1; l++) {
			if      (termnum==1)  weight1[l]=-wcl.tg(l)/dcl.tt(l);
			else if (termnum==2)  weight1[l]=wcl.tg(l)/dcl.tt(l);
			else if (termnum==3)  weight1[l]=wcl.tg(l)*(l*(l+1.))/dcl.tt(l);
			else if (termnum==4)  weight1[l]=-wcl.tg(l)*sgn(l)/dcl.tt(l);
			else if (termnum==5)  weight1[l]=wcl.tg(l)*sgn(l)/dcl.tt(l);
			else if (termnum==6)  weight1[l]=wcl.tg(l)*sgn(l)*(l*(l+1.))/dcl.tt(l);
			else if (termnum==7)  weight1[l]=-(l*(l+1.))/dcl.tt(l);
			else if (termnum==8)  weight1[l]=1./dcl.tt(l);
			else if (termnum==9)  weight1[l]=1./dcl.tt(l);
			else if (termnum==10) weight1[l]=-(l*(l+1.))*sgn(l)/dcl.tt(l);
			else if (termnum==11) weight1[l]=sgn(l)/dcl.tt(l);
			else if (termnum==12) weight1[l]=sgn(l)*1./dcl.tt(l);
		
			if (weight1[l]!=weight1[l]) weight1[l]=.0;
		}
		#pragma omp parallel for
		for (size_t l=lminCMB2; l<lmaxCMB2+1; l++) {
			if      (termnum==1)  weight2[l]=(l*(l+1.))/dcl.gg(l);
			else if (termnum==2)  weight2[l]=1./dcl.gg(l);
			else if (termnum==3)  weight2[l]=1./dcl.gg(l);
			else if (termnum==4)  weight2[l]=(l*(l+1.))*sgn(l)/dcl.gg(l);
			else if (termnum==5)  weight2[l]=sgn(l)/dcl.gg(l);
			else if (termnum==6)  weight2[l]=1.*sgn(l)/dcl.gg(l);
			else if (termnum==7)  weight2[l]=wcl.tg(l)/dcl.gg(l);
			else if (termnum==8)  weight2[l]=wcl.tg(l)/dcl.gg(l);
			else if (termnum==9)  weight2[l]=(l*(l+1.))*wcl.tg(l)/dcl.gg(l);
			else if (termnum==10) weight2[l]=sgn(l)*wcl.tg(l)/dcl.gg(l);
			else if (termnum==11) weight2[l]=sgn(l)*wcl.tg(l)/dcl.gg(l);
			else if (termnum==12) weight2[l]=(l*(l+1.))*sgn(l)*wcl.tg(l)/dcl.gg(l);
		
			if (weight2[l]!=weight2[l]) weight2[l]=.0;
		}
		if (termnum<=6) spin=2;
		else spin=0;
	}
	
	else if (type==ee) { 
		#pragma omp parallel for
		for (size_t l=lminCMB1; l<lmaxCMB1+1; l++) {
			if (termnum==1)       weight1[l]=-wcl.gg(l)/dcl.gg(l);
			else if (termnum==2)  weight1[l]=wcl.gg(l)/dcl.gg(l);
			else if (termnum==3)  weight1[l]=wcl.gg(l)*(l*(l+1.))/dcl.gg(l);
			else if (termnum==4)  weight1[l]=-wcl.gg(l)*sgn(l)/dcl.gg(l);
			else if (termnum==5)  weight1[l]=wcl.gg(l)*sgn(l)/dcl.gg(l);
			else if (termnum==6)  weight1[l]=wcl.gg(l)*(l*(l+1.))*sgn(l)/dcl.gg(l);
			else if (termnum==7)  weight1[l]=-(l*(l+1.))/dcl.gg(l);
			else if (termnum==8)  weight1[l]=1./dcl.gg(l);
			else if (termnum==9)  weight1[l]=1./dcl.gg(l);
			else if (termnum==10) weight1[l]=-sgn(l)*(l*(l+1.))/dcl.gg(l);
			else if (termnum==11) weight1[l]=sgn(l)*1./dcl.gg(l);
			else if (termnum==12) weight1[l]=sgn(l)*1./dcl.gg(l);
			
			if (weight1[l]!=weight1[l]) weight1[l]=.0;
		}
		#pragma omp parallel for
		for (size_t l=lminCMB2; l<lmaxCMB2+1; l++) {
			if      (termnum==1)  weight2[l]=(l*(l+1.))/dcl.gg(l);
			else if (termnum==2)  weight2[l]=1./dcl.gg(l);
			else if (termnum==3)  weight2[l]=1./dcl.gg(l);
			else if (termnum==4)  weight2[l]=(l*(l+1.))*sgn(l)/dcl.gg(l);
			else if (termnum==5)  weight2[l]=sgn(l)*1./dcl.gg(l);
			else if (termnum==6)  weight2[l]=sgn(l)*1./dcl.gg(l);
			else if (termnum==7)  weight2[l]=wcl.gg(l)/dcl.gg(l);
			else if (termnum==8)  weight2[l]=wcl.gg(l)/dcl.gg(l);
			else if (termnum==9)  weight2[l]=wcl.gg(l)*(l*(l+1.))/dcl.gg(l);
			else if (termnum==10) weight2[l]=wcl.gg(l)*sgn(l)/dcl.gg(l);
			else if (termnum==11) weight2[l]=wcl.gg(l)*sgn(l)/dcl.gg(l);
			else if (termnum==12) weight2[l]=wcl.gg(l)*(l*(l+1.))*sgn(l)/dcl.gg(l);
			
			if (weight2[l]!=weight2[l]) weight2[l]=.0;
		}
		spin=2;
	}
	
	else if (type==tb) { 
		#pragma omp parallel for
		for (size_t l=lminCMB1; l<lmaxCMB1+1; l++) {
			if      (termnum==1) weight1[l]=wcl.tg(l)/dcl.tt(l);
			else if (termnum==2) weight1[l]=-wcl.tg(l)/dcl.tt(l);
			else if (termnum==3) weight1[l]=-wcl.tg(l)*(l*(l+1.))/dcl.tt(l);
			else if (termnum==4) weight1[l]=-wcl.tg(l)*sgn(l)/dcl.tt(l);
			else if (termnum==5) weight1[l]=wcl.tg(l)*sgn(l)/dcl.tt(l);
			else if (termnum==6) weight1[l]=wcl.tg(l)*(l*(l+1.))*sgn(l)/dcl.tt(l);
			
			if (weight1[l]!=weight1[l]) weight1[l]=.0;
		}
		#pragma omp parallel for
		for (size_t l=lminCMB2; l<lmaxCMB2+1; l++) {
			if      (termnum==1) weight2[l]=(l*(l+1.))/dcl.cc(l);
			else if (termnum==2) weight2[l]=1./dcl.cc(l);
			else if (termnum==3) weight2[l]=1./dcl.cc(l);
			else if (termnum==4) weight2[l]=(l*(l+1.))*sgn(l)/dcl.cc(l);
			else if (termnum==5) weight2[l]=sgn(l)*1./dcl.cc(l);
			else if (termnum==6) weight2[l]=sgn(l)*1./dcl.cc(l);
			
			if (weight2[l]!=weight2[l]) weight2[l]=.0;
		}
		spin=2;
	}
	
	else if (type==eb) { 
		#pragma omp parallel for
		for (size_t l=lminCMB1; l<lmaxCMB1+1; l++) {
			if      (termnum==1) weight1[l]=wcl.gg(l)/dcl.gg(l);
			else if (termnum==2) weight1[l]=-wcl.gg(l)/dcl.gg(l);
			else if (termnum==3) weight1[l]=-wcl.gg(l)*(l*(l+1.))/dcl.gg(l);
			else if (termnum==4) weight1[l]=-wcl.gg(l)*sgn(l)/dcl.gg(l);
			else if (termnum==5) weight1[l]=wcl.gg(l)*sgn(l)/dcl.gg(l);
			else if (termnum==6) weight1[l]=wcl.gg(l)*(l*(l+1.))*sgn(l)/dcl.gg(l);
			
			if (weight1[l]!=weight1[l]) weight1[l]=.0;
		}
		#pragma omp parallel for
		for (size_t l=lminCMB2; l<lmaxCMB2+1; l++) {
			if      (termnum==1) weight2[l]=(l*(l+1.))/dcl.cc(l);
			else if (termnum==2) weight2[l]=1./dcl.cc(l);
			else if (termnum==3) weight2[l]=1./dcl.cc(l);
			else if (termnum==4) weight2[l]=(l*(l+1.))*sgn(l)/dcl.cc(l);
			else if (termnum==5) weight2[l]=sgn(l)*1./dcl.cc(l);
			else if (termnum==6) weight2[l]=sgn(l)*1./dcl.cc(l);
			
			if (weight2[l]!=weight2[l]) weight2[l]=.0;
		}
		spin=2;
	}
	
	else if (type==bb) {
		#pragma omp parallel for
		for (size_t l=lminCMB1; l<lmaxCMB1+1; l++) {
			if      (termnum==1)  weight1[l]=-wcl.cc(l)/dcl.cc(l);
			else if (termnum==2)  weight1[l]=wcl.cc(l)/dcl.cc(l);
			else if (termnum==3)  weight1[l]=wcl.cc(l)*(l*(l+1.))/dcl.cc(l);
			else if (termnum==4)  weight1[l]=-wcl.cc(l)*sgn(l)/dcl.cc(l);
			else if (termnum==5)  weight1[l]=wcl.cc(l)*sgn(l)/dcl.cc(l);
			else if (termnum==6)  weight1[l]=wcl.cc(l)*(l*(l+1.))*sgn(l)/dcl.cc(l);
			else if (termnum==7)  weight1[l]=-(l*(l+1.))/dcl.cc(l);
			else if (termnum==8)  weight1[l]=1./dcl.cc(l);
			else if (termnum==9)  weight1[l]=1./dcl.cc(l);
			else if (termnum==10) weight1[l]=-sgn(l)*(l*(l+1.))/dcl.cc(l);
			else if (termnum==11) weight1[l]=sgn(l)*1./dcl.cc(l);
			else if (termnum==12) weight1[l]=sgn(l)*1./dcl.cc(l);
			
			if (weight1[l]!=weight1[l]) weight1[l]=.0;
		}
		#pragma omp parallel for
		for (size_t l=lminCMB2; l<lmaxCMB2+1; l++) {
			if      (termnum==1)  weight2[l]=(l*(l+1.))/dcl.cc(l);
			else if (termnum==2)  weight2[l]=1./dcl.cc(l);
			else if (termnum==3)  weight2[l]=1./dcl.cc(l);
			else if (termnum==4)  weight2[l]=(l*(l+1.))*sgn(l)/dcl.cc(l);
			else if (termnum==5)  weight2[l]=sgn(l)*1./dcl.cc(l);
			else if (termnum==6)  weight2[l]=sgn(l)*1./dcl.cc(l);
			else if (termnum==7)  weight2[l]=wcl.cc(l)/dcl.cc(l);
			else if (termnum==8)  weight2[l]=wcl.cc(l)/dcl.cc(l);
			else if (termnum==9)  weight2[l]=wcl.cc(l)*(l*(l+1.))/dcl.cc(l);
			else if (termnum==10) weight2[l]=wcl.cc(l)*sgn(l)/dcl.cc(l);
			else if (termnum==11) weight2[l]=wcl.cc(l)*sgn(l)/dcl.cc(l);
			else if (termnum==12) weight2[l]=wcl.cc(l)*(l*(l+1.))*sgn(l)/dcl.cc(l);
			
			if (weight2[l]!=weight2[l]) weight2[l]=.0;
		}
		spin=2;
	}
	
	#pragma omp parallel for
	for (size_t m=0; m<=lmaxCMB1; ++m) {
		for (size_t l=m; l<=lmaxCMB1; ++l) {
			alm1(l,m)=weight1[l]*alm1in(l,m);
		}
	} 
	#pragma omp parallel for
	for (size_t m=0; m<=lmaxCMB2; ++m) {
		for (size_t l=m; l<=lmaxCMB2; ++l) {
			alm2(l,m)=weight2[l]*alm2in(l,m);
		}
	} 
	
	Alm< xcomplex< double > > almout(lmax,lmax);
	fast_kernel(alm1,alm2,almout,spin,nside,weight);

	if (termnum==2 || termnum==5 || termnum==8 || termnum==11) almout.ScaleL(lsqr);
	if (termnum==4 || termnum==5 || termnum==6 || termnum==10 || termnum==11 || termnum==12) almout.ScaleL(sgnL);

	if (type==tt) almout.Scale(1.);
	else if (type==te) almout.Scale(1.);
	else if (type==ee) almout.Scale(.5);
	else if (type==tb) almout.Scale(1.*complex_i);
	else if (type==eb) almout.Scale(1.*complex_i);
	else if (type==bb) almout.Scale(.5);
	
	almP.Add(almout);
	
	if(PyErr_CheckSignals() == -1) {
		throw invalid_argument( "Keyboard interrupt" );
	}
}


