#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream> 

#include <healpix_map.h>
#include <alm.h>
#include <alm_healpix_tools.h>
#include <sharp_cxx.h>
#include <fitshandle.h>

using namespace std;

xcomplex< double > compi(0.0,1.0);	
	
void purealm_spin0_update(Alm<xcomplex<double> > & almG, Alm<xcomplex<double> > & almC, Alm<xcomplex<double> > & localalmG, Alm<xcomplex<double> > & localalmC, bool pureE) {
	int lmax=almG.Lmax();
	
	arr<double> sqrtfact;
	sqrtfact.alloc(lmax+1);
	sqrtfact.fill(.0);
	for (int l=2; l<lmax+1; l++) {
		sqrtfact[l]=1.0/sqrt((double)((l-1.0)*l*(l+1.0)*(l+2.0)));
	}
	
	for (int m=0; m<=lmax; ++m) {
		for (int l=m; l<=lmax; ++l) {
			if (pureE) almG(l,m)+=sqrtfact[l]*localalmG(l,m);
			almC(l,m)+=sqrtfact[l]*localalmC(l,m);
		}
	}
}

void purealm_spin1_update(Alm<xcomplex<double> > & almG, Alm<xcomplex<double> > & almC, Alm<xcomplex<double> > & localalmG, Alm<xcomplex<double> > & localalmC, bool pureE) {
	int lmax=almG.Lmax();
	
	arr<double> sqrtfact;
	sqrtfact.alloc(lmax+1);
	sqrtfact.fill(.0);
	for (int l=2; l<lmax+1; l++) {
		sqrtfact[l]=2.0/sqrt((double)((l-1.0)*(l+2.0)));
	}
	
	for (int m=0; m<=lmax; ++m) {
		for (int l=m; l<=lmax; ++l) {
			if (pureE) almG(l,m)+=sqrtfact[l]*localalmG(l,m);
			almC(l,m)+=sqrtfact[l]*localalmC(l,m)*compi;
		}
	}
}

void purealm_spin2_update(Alm<xcomplex<double> > & almG, Alm<xcomplex<double> > & almC, Alm<xcomplex<double> > & localalmG, Alm<xcomplex<double> > & localalmC) {
	int lmax=almG.Lmax();
	for (int m=0; m<=lmax; ++m) {
		for (int l=m; l<=lmax; ++l) {
			almG(l,m)+=localalmG(l,m);
			almC(l,m)+=localalmC(l,m);
			if (l<2) {almG(l,m)=.0; almC(l,m)=.0;}
		}
	}
}
	
void wlm_scalar2vector(Alm<xcomplex<double> > &wlm) {
	int lmax=wlm.Lmax();
	
	wlm(0,0)=0.;
	
	arr<xcomplex<double>> sqrtfact;
	sqrtfact.alloc(lmax+1);
	for (int l=0; l<lmax+1; l++) {
		sqrtfact[l]=sqrt((double)(l*(l+1.)));
	}
	
	wlm.ScaleL(sqrtfact);
}
   
void wlm_vector2tensor(Alm<xcomplex<double> > &wlm) {
	int lmax=wlm.Lmax();
	
	wlm(0,0)=0.;
	
	arr<double> sqrtfact;
	sqrtfact.alloc(lmax+1);
	for (int l=0; l<lmax+1; l++) {
		sqrtfact[l]=sqrt((double)((l-1.)*(l+2.)));
	}
	
	wlm.ScaleL(sqrtfact);
}

void combine_mask_window(Healpix_Map<double> &W, const Healpix_Map<double> &mask) {
	for (int i=0; i< W.Npix(); i++) {
		W[i]*=mask[i];
	}
}

void apodize_maps_complex(Healpix_Map<double> &Wre, Healpix_Map<double> &Wim, Healpix_Map<double> &mapU, Healpix_Map<double> &mapQ) {
		double temp=0;
		for (int i=0; i< Wre.Npix(); i++) {
			temp=Wre[i];
			Wre[i]=temp*mapQ[i]+Wim[i]*mapU[i];
			Wim[i]=temp*mapU[i]-Wim[i]*mapQ[i];
		}
}

void map2purealm
	(Healpix_Map<double> &mapT, 
	Healpix_Map<double> &mapQ,
	Healpix_Map<double> &mapU,
	const Healpix_Map<double> &W0,
	const Healpix_Map<double> &mask,
	Alm<xcomplex<double> > &almT,
	Alm<xcomplex<double> > &almG,
	Alm<xcomplex<double> > &almC,
	const arr<double> &weight,
	bool pureE)
	{
		planck_assert (mapT.Scheme()==RING,
		"map2purealm: maps must be in RING scheme");
		planck_assert (mapT.conformable(mapQ) && mapT.conformable(mapU),
		"map2purealm: maps are not conformable");
		planck_assert (almT.conformable(almG) && almT.conformable(almC),
		"map2purealm: a_lm are not conformable");
		planck_assert (int(weight.size())>=2*mapT.Nside(),
		"map2purealm: weight array has too few entries");
		planck_assert (mapT.fullyDefined()&&mapQ.fullyDefined()&&mapU.fullyDefined(),
		"map contains undefined pixels");

		almG.SetToZero();
		almC.SetToZero();
		
		int nside=W0.Nside();
		int lmax=almT.Lmax();
		int mmax=almT.Mmax();
		
		Alm< xcomplex< double > > local_wlm(lmax,mmax), almZ(lmax,mmax);
		
		map2alm(W0,local_wlm,weight,false);
		
		Healpix_Map<double> Wre, Wim;
        Wre.SetNside(nside,RING);
        Wim.SetNside(nside,RING);
		
		for (int i=0; i< Wre.Npix(); i++) {
			Wre[i]=W0[i]*mapQ[i];
			Wim[i]=W0[i]*mapU[i];
		}
		
		Alm< xcomplex< double > > local_almG(lmax,mmax),local_almC(lmax,mmax);
		
		sharp_cxxjob<double> job;
		job.set_weighted_Healpix_geometry (nside,&weight[0]);
		job.set_triangular_alm_info (lmax,mmax);
		job.map2alm_spin(&Wre[0],&Wim[0],&local_almG(0,0),&local_almC(0,0),2,false);
		
		purealm_spin2_update(almG,almC,local_almG,local_almC);
		
		wlm_scalar2vector(local_wlm);
		
		job.alm2map_spin(&local_wlm(0,0),&almZ(0,0),&Wre[0],&Wim[0],1,false);
		
		combine_mask_window(Wre,mask);
		combine_mask_window(Wim,mask);
		
		apodize_maps_complex(Wre,Wim,mapQ,mapU);
		
		job.map2alm_spin(&Wre[0],&Wim[0],&local_almG(0,0),&local_almC(0,0),1,false);

		purealm_spin1_update(almG,almC,local_almG,local_almC,pureE);
		
		wlm_vector2tensor(local_wlm);
		
		job.alm2map_spin(&local_wlm(0,0),&almZ(0,0),&Wre[0],&Wim[0],2,false);
		
		combine_mask_window(Wre,mask);
		combine_mask_window(Wim,mask);
		
		apodize_maps_complex(Wre,Wim,mapQ,mapU);
		
		job.map2alm_spin(&Wre[0],&Wim[0],&local_almG(0,0),&local_almC(0,0),0,false);

		purealm_spin0_update(almG,almC,local_almG,local_almC,pureE);
		
		job.map2alm(&mapT[0], &almT(0,0), false);
	}