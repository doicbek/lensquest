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

#include "_lensquest_cxx.h"
#include "_lensquest_cxx_old.h"

#include "_cxx_healpix.cpp"
#include "_cxx_lens.cpp"
#include "_cxx_amp.cpp"
#include "_cxx_dbeta.cpp"
#include "_cxx_rot.cpp"
#include "_cxx_spinflip.cpp"
#include "_cxx_monoleak.cpp"
#include "_cxx_phodir.cpp"
#include "_cxx_dipleak.cpp"
#include "_cxx_quadleak.cpp"



void est_grad(Alm< xcomplex< double > > &alm1, Alm< xcomplex< double > > &alm2, std::string stype, Alm< xcomplex< double > > &almG, Alm< xcomplex< double > > &almC, PowSpec& wcl, PowSpec& dcl, int lmin, int lminCMB1, int lminCMB2,  int lmaxCMB1, int lmaxCMB2, int nside) {
	size_t lmax=almG.Lmax();

	almG.SetToZero();
	almC.SetToZero();
	int type = string2esttype(stype);
	
	arr<double> weight;
	weight.alloc(2*nside);
	weight.fill(1.0);
	
	switch(type) {
		case tt: lensTT(alm1, alm2, almG, almC, wcl, nside, weight); break;
		case te: lensTE(alm1, alm2, almG, almC, wcl, nside, weight); break;
		case tb: lensTB(alm1, alm2, almG, almC, wcl, nside, weight); break;
		case ee: lensEE(alm1, alm2, almG, almC, wcl, nside, weight); break;
		case eb: lensEB(alm1, alm2, almG, almC, wcl, nside, weight); break;
		case bb: lensBB(alm1, alm2, almG, almC, wcl, nside, weight); break;
	}
}

void est_amp(Alm< xcomplex< double > > &alm1, Alm< xcomplex< double > > &alm2, std::string stype, Alm< xcomplex< double > > &almG, PowSpec& wcl, PowSpec& dcl, int lmin, int lminCMB1, int lminCMB2,  int lmaxCMB1, int lmaxCMB2, int nside) {
	size_t lmax=almG.Lmax();

	Alm< xcomplex< double > > almC(lmax,lmax);
	almG.SetToZero();
	int type = string2esttype(stype);
	
	arr<double> weight;
	weight.alloc(2*nside);
	weight.fill(1.0);
	
	switch(type) {
		case tt: ampTT(alm1, alm2, almG, almC, wcl, nside, weight); break;
		case te: ampTE(alm1, alm2, almG, almC, wcl, nside, weight); break;
		case tb: ampTB(alm1, alm2, almG, almC, wcl, nside, weight); break;
		case ee: ampEE(alm1, alm2, almG, almC, wcl, nside, weight); break;
		case eb: ampEB(alm1, alm2, almG, almC, wcl, nside, weight); break;
		case bb: ampBB(alm1, alm2, almG, almC, wcl, nside, weight); break;
	}
}

void est_dbeta(Alm< xcomplex< double > > &alm1, Alm< xcomplex< double > > &alm2, std::string stype, Alm< xcomplex< double > > &almG, PowSpec& wcl, PowSpec& dcl, std::vector<double>& R1, std::vector<double>& R2, int lmin, int lminCMB1, int lminCMB2,  int lmaxCMB1, int lmaxCMB2, int nside) {
	size_t lmax=almG.Lmax();

	Alm< xcomplex< double > > almC(lmax,lmax);
	almG.SetToZero();
	int type = string2esttype(stype);
	
	arr<double> weight;
	weight.alloc(2*nside);
	weight.fill(1.0);
	
	switch(type) {
		case tt: dbetaTT(alm1, alm2, almG, almC, wcl, R1, R2, nside, weight); break;
		case te: dbetaTE(alm1, alm2, almG, almC, wcl, R1, R2, nside, weight); break;
		case tb: dbetaTB(alm1, alm2, almG, almC, wcl, R1, R2, nside, weight); break;
		case ee: dbetaEE(alm1, alm2, almG, almC, wcl, R1, R2, nside, weight); break;
		case eb: dbetaEB(alm1, alm2, almG, almC, wcl, R1, R2, nside, weight); break;
		case bb: dbetaBB(alm1, alm2, almG, almC, wcl, R1, R2, nside, weight); break;
	}
}

void est_rot(Alm< xcomplex< double > > &alm1, Alm< xcomplex< double > > &alm2, std::string stype, Alm< xcomplex< double > > &almG, PowSpec& wcl, PowSpec& dcl, int lmin, int lminCMB1, int lminCMB2,  int lmaxCMB1, int lmaxCMB2, int nside) {
	size_t lmax=almG.Lmax();

	Alm< xcomplex< double > > almC(lmax,lmax);
	almG.SetToZero();
	int type = string2esttype(stype);
	
	arr<double> weight;
	weight.alloc(2*nside);
	weight.fill(1.0);
	
	switch(type) {
		case te: rotTE(alm1, alm2, almG, almC, wcl, nside, weight); break;
		case tb: rotTB(alm1, alm2, almG, almC, wcl, nside, weight); break;
		case ee: rotEE(alm1, alm2, almG, almC, wcl, nside, weight); break;
		case eb: rotEB(alm1, alm2, almG, almC, wcl, nside, weight); break;
	}
}

void est_spinflip(Alm< xcomplex< double > > &alm1, Alm< xcomplex< double > > &alm2, std::string stype, Alm< xcomplex< double > > &almG,  Alm< xcomplex< double > > &almC, PowSpec& wcl, PowSpec& dcl, int lmin, int lminCMB1, int lminCMB2,  int lmaxCMB1, int lmaxCMB2, int nside) {
	size_t lmax=almG.Lmax();

	almG.SetToZero();
	almC.SetToZero();
	int type = string2esttype(stype);
	
	arr<double> weight;
	weight.alloc(2*nside);
	weight.fill(1.0);
	
	switch(type) {
		case te: spiTE(alm1, alm2, almG, almC, wcl, nside, weight); break;
		case tb: spiTB(alm1, alm2, almG, almC, wcl, nside, weight); break;
		case ee: spiEE(alm1, alm2, almG, almC, wcl, nside, weight); break;
		case eb: spiEB(alm1, alm2, almG, almC, wcl, nside, weight); break;
	}
}

void est_monoleak(Alm< xcomplex< double > > &alm1, Alm< xcomplex< double > > &alm2, std::string stype, Alm< xcomplex< double > > &almG,  Alm< xcomplex< double > > &almC, PowSpec& wcl, PowSpec& dcl, int lmin, int lminCMB1, int lminCMB2,  int lmaxCMB1, int lmaxCMB2, int nside) {
	size_t lmax=almG.Lmax();

	almG.SetToZero();
	almC.SetToZero();
	int type = string2esttype(stype);
	
	arr<double> weight;
	weight.alloc(2*nside);
	weight.fill(1.0);
	
	switch(type) {
		case te: lmoTE(alm1, alm2, almG, almC, wcl, nside, weight); break;
		case tb: lmoTB(alm1, alm2, almG, almC, wcl, nside, weight); break;
		case ee: lmoEE(alm1, alm2, almG, almC, wcl, nside, weight); break;
		case eb: lmoEB(alm1, alm2, almG, almC, wcl, nside, weight); break;
	}
}

void est_phodir(Alm< xcomplex< double > > &alm1, Alm< xcomplex< double > > &alm2, std::string stype, Alm< xcomplex< double > > &almG,  Alm< xcomplex< double > > &almC, PowSpec& wcl, PowSpec& dcl, int lmin, int lminCMB1, int lminCMB2,  int lmaxCMB1, int lmaxCMB2, int nside) {
	size_t lmax=almG.Lmax();

	almG.SetToZero();
	almC.SetToZero();
	int type = string2esttype(stype);
	
	arr<double> weight;
	weight.alloc(2*nside);
	weight.fill(1.0);
	
	switch(type) {
		case te: dirTE(alm1, alm2, almG, almC, wcl, nside, weight); break;
		case tb: dirTB(alm1, alm2, almG, almC, wcl, nside, weight); break;
		case ee: dirEE(alm1, alm2, almG, almC, wcl, nside, weight); break;
		case eb: dirEB(alm1, alm2, almG, almC, wcl, nside, weight); break;
	}
}

void est_dipleak(Alm< xcomplex< double > > &alm1, Alm< xcomplex< double > > &alm2, std::string stype, Alm< xcomplex< double > > &almG,  Alm< xcomplex< double > > &almC, PowSpec& wcl, PowSpec& dcl, int lmin, int lminCMB1, int lminCMB2,  int lmaxCMB1, int lmaxCMB2, int nside) {
	size_t lmax=almG.Lmax();

	almG.SetToZero();
	almC.SetToZero();
	int type = string2esttype(stype);
	
	arr<double> weight;
	weight.alloc(2*nside);
	weight.fill(1.0);
	
	switch(type) {
		case te: ldiTE(alm1, alm2, almG, almC, wcl, nside, weight); break;
		case tb: ldiTB(alm1, alm2, almG, almC, wcl, nside, weight); break;
		case ee: ldiEE(alm1, alm2, almG, almC, wcl, nside, weight); break;
		case eb: ldiEB(alm1, alm2, almG, almC, wcl, nside, weight); break;
	}
}

void est_quadleak(Alm< xcomplex< double > > &alm1, Alm< xcomplex< double > > &alm2, std::string stype, Alm< xcomplex< double > > &almG,  Alm< xcomplex< double > > &almC, PowSpec& wcl, PowSpec& dcl, int lmin, int lminCMB1, int lminCMB2,  int lmaxCMB1, int lmaxCMB2, int nside) {
	size_t lmax=almG.Lmax();

	almG.SetToZero();
	almC.SetToZero();
	int type = string2esttype(stype);
	
	arr<double> weight;
	weight.alloc(2*nside);
	weight.fill(1.0);
	
	switch(type) {
		case te: lquTE(alm1, alm2, almG, almC, wcl, nside, weight); break;
		case tb: lquTB(alm1, alm2, almG, almC, wcl, nside, weight); break;
		case ee: lquEE(alm1, alm2, almG, almC, wcl, nside, weight); break;
		case eb: lquEB(alm1, alm2, almG, almC, wcl, nside, weight); break;
	}
}


void computef_systa(std::vector< std::vector< std::vector<xcomplex< double >> > >& f, size_t L, PowSpec& wcl, size_t lminCMB, size_t lmaxCMB){
	
	std::vector< std::vector<double> > F;
	
	F=std::vector< std::vector<double> >(lmaxCMB+1, std::vector<double>(lmaxCMB+1,0.0));
	
	compF_a(F, L, lmaxCMB+1);
	
	#pragma omp parallel for
	for (size_t l1=lminCMB;l1<lmaxCMB+1;l1++) {
		for (size_t l3=lminCMB;l3<lmaxCMB+1;l3++) {
			if ((l1+L+l3)%2==0) {
				f[0][l1][l3]=wcl.tg(l1)*F[l3][l1];
				f[1][l1][l3]=wcl.gg(l1)*F[l3][l1]+wcl.gg(l3)*F[l1][l3];
				f[2][l1][l3]=0.0; 
				f[3][l1][l3]=0.0;
			}
			else {
				f[0][l1][l3]=0.0; 
				f[1][l1][l3]=0.0; 
				f[2][l1][l3]=-wcl.tg(l1)*F[l3][l1];
				f[3][l1][l3]=-wcl.gg(l1)*F[l3][l1]+wcl.cc(l3)*F[l1][l3];
			}
		}
	}
}

void computef_systomega(std::vector< std::vector< std::vector<xcomplex< double >> > >& f, size_t L, PowSpec& wcl, size_t lminCMB, size_t lmaxCMB){
	
	std::vector< std::vector<double> > F;
	
	F=std::vector< std::vector<double> >(lmaxCMB+1, std::vector<double>(lmaxCMB+1,0.0));
	
	compF_a(F, L, lmaxCMB+1);
	
	#pragma omp parallel for
	for (size_t l1=lminCMB;l1<lmaxCMB+1;l1++) {
		for (size_t l3=lminCMB;l3<lmaxCMB+1;l3++) {
			if ((l1+L+l3)%2!=0) {
				f[0][l1][l3]=-2*wcl.tg(l1)*F[l3][l1];
				f[1][l1][l3]=-2*wcl.gg(l1)*F[l3][l1]+2*wcl.gg(l3)*F[l1][l3];
				f[2][l1][l3]=0.0; 
				f[3][l1][l3]=0.0;
			}
			else {
				f[0][l1][l3]=0.0; 
				f[1][l1][l3]=0.0; 
				f[2][l1][l3]=2*wcl.tg(l1)*F[l3][l1];
				f[3][l1][l3]=2*wcl.gg(l1)*F[l3][l1]-2*wcl.cc(l3)*F[l1][l3];
			}
		}
	}
}

void computef_systfa(std::vector< std::vector< std::vector<xcomplex< double >> > >& f, size_t L, PowSpec& wcl, size_t lminCMB, size_t lmaxCMB){
	
	std::vector< std::vector<double> > F;
	
	F=std::vector< std::vector<double> >(lmaxCMB+1, std::vector<double>(lmaxCMB+1,0.0));
	
	compF_f(F, L, lmaxCMB+1);
	
	#pragma omp parallel for
	for (size_t l1=lminCMB;l1<lmaxCMB+1;l1++) {
		for (size_t l3=lminCMB;l3<lmaxCMB+1;l3++) {
			if ((l1+L+l3)%2==0) {
				f[0][l1][l3]=wcl.tg(l1)*complex_i*F[l3][l1];
				f[1][l1][l3]=wcl.gg(l1)*complex_i*F[l3][l1]+wcl.gg(l3)*F[l1][l3];
				f[2][l1][l3]=0.0; 
				f[3][l1][l3]=0.0;
			}
			else {
				f[0][l1][l3]=0.0; 
				f[1][l1][l3]=0.0; 
				f[2][l1][l3]=-wcl.tg(l1)*complex_i*F[l3][l1];
				f[3][l1][l3]=-wcl.gg(l1)*complex_i*F[l3][l1]+wcl.cc(l3)*complex_i*F[l1][l3];
			}
		}
	}
}

void computef_systfb(std::vector< std::vector< std::vector<xcomplex< double >> > >& f, size_t L, PowSpec& wcl, size_t lminCMB, size_t lmaxCMB){
	
	std::vector< std::vector<double> > F;
	
	F=std::vector< std::vector<double> >(lmaxCMB+1, std::vector<double>(lmaxCMB+1,0.0));
	
	compF_f(F, L, lmaxCMB+1);
	
	#pragma omp parallel for
	for (size_t l1=lminCMB;l1<lmaxCMB+1;l1++) {
		for (size_t l3=lminCMB;l3<lmaxCMB+1;l3++) {
			if ((l1+L+l3)%2!=0) {
				f[0][l1][l3]=-wcl.tg(l1)*complex_i*F[l3][l1];
				f[1][l1][l3]=-wcl.gg(l1)*complex_i*F[l3][l1]+wcl.gg(l3)*complex_i*F[l1][l3];
				f[2][l1][l3]=0.0; 
				f[3][l1][l3]=0.0;
			}
			else {
				f[0][l1][l3]=0.0; 
				f[1][l1][l3]=0.0; 
				f[2][l1][l3]=wcl.tg(l1)*complex_i*F[l3][l1];
				f[3][l1][l3]=wcl.gg(l1)*complex_i*F[l3][l1]-wcl.cc(l3)*complex_i*F[l1][l3];
			}
		}
	}
}

void computef_systgammaa(std::vector< std::vector< std::vector<xcomplex< double >> > >& f, size_t L, PowSpec& wcl, size_t lminCMB, size_t lmaxCMB){
	
	std::vector< std::vector<double> > F;
	
	F=std::vector< std::vector<double> >(lmaxCMB+1, std::vector<double>(lmaxCMB+1,0.0));
	
	compF_gamma(F, L, lmaxCMB+1);
	
	#pragma omp parallel for
	for (size_t l1=lminCMB;l1<lmaxCMB+1;l1++) {
		for (size_t l3=lminCMB;l3<lmaxCMB+1;l3++) {
			if ((l1+L+l3)%2==0) {
				f[0][l1][l3]=wcl.tt(l1)*F[l3][l1];
				f[1][l1][l3]=wcl.tg(l1)*F[l3][l1]+wcl.tg(l3)*F[l1][l3];
				f[2][l1][l3]=0.0; 
				f[3][l1][l3]=0.0;
			}
			else {
				f[0][l1][l3]=0.0; 
				f[1][l1][l3]=0.0; 
				f[2][l1][l3]=-wcl.tt(l1)*F[l3][l1];
				f[3][l1][l3]=-wcl.tg(l1)*F[l3][l1];
			}
		}
	}
}

void computef_systgammab(std::vector< std::vector< std::vector<xcomplex< double >> > >& f, size_t L, PowSpec& wcl, size_t lminCMB, size_t lmaxCMB){
	
	std::vector< std::vector<double> > F;
	
	F=std::vector< std::vector<double> >(lmaxCMB+1, std::vector<double>(lmaxCMB+1,0.0));
	
	compF_gamma(F, L, lmaxCMB+1);
	
	#pragma omp parallel for
	for (size_t l1=lminCMB;l1<lmaxCMB+1;l1++) {
		for (size_t l3=lminCMB;l3<lmaxCMB+1;l3++) {
			if ((l1+L+l3)%2!=0) {
				f[0][l1][l3]=-wcl.tt(l1)*F[l3][l1];
				f[1][l1][l3]=-wcl.tg(l1)*F[l3][l1]+wcl.tg(l3)*F[l1][l3];
				f[2][l1][l3]=0.0; 
				f[3][l1][l3]=0.0;
			}
			else {
				f[0][l1][l3]=0.0; 
				f[1][l1][l3]=0.0; 
				f[2][l1][l3]=wcl.tt(l1)*F[l3][l1];
				f[3][l1][l3]=wcl.tg(l1)*F[l3][l1];
			}
		}
	}
}

void computef_systpa(std::vector< std::vector< std::vector<xcomplex< double >> > >& f, size_t L, PowSpec& wcl, size_t lminCMB, size_t lmaxCMB){
	
	std::vector< std::vector<double> > Fa, Fb;
	
	Fa=std::vector< std::vector<double> >(lmaxCMB+1, std::vector<double>(lmaxCMB+1,0.0));
	Fb=std::vector< std::vector<double> >(lmaxCMB+1, std::vector<double>(lmaxCMB+1,0.0));
	
	compF_pa(Fa, L, lmaxCMB+1);
	compF_pb(Fb, L, lmaxCMB+1);
	
	#pragma omp parallel for
	for (size_t l1=lminCMB;l1<lmaxCMB+1;l1++) {
		for (size_t l3=lminCMB;l3<lmaxCMB+1;l3++) {
			if ((l1+L+l3)%2!=0) {
				f[0][l1][l3]=0;
				f[1][l1][l3]=0;
				f[2][l1][l3]=.5*wcl.tg(l1)*(Fb[l3][l1]-Fa[l3][l1]);
				f[3][l1][l3]=.5*wcl.gg(l1)*(Fb[l3][l1]-Fa[l3][l1])+.5*wcl.cc(l3)*(Fb[l1][l3]-Fa[l1][l3]);
			}
			else {
				f[0][l1][l3]=.5*wcl.tg(l1)*(Fa[l3][l1]+Fb[l3][l1]);
				f[1][l1][l3]=.5*wcl.gg(l1)*(Fa[l3][l1]+Fb[l3][l1])+.5*wcl.gg(l3)*(Fa[l1][l3]+Fb[l1][l3]);
				f[2][l1][l3]=0;
				f[3][l1][l3]=0;
			}
		}
	}
}

void computef_systpb(std::vector< std::vector< std::vector<xcomplex< double >> > >& f, size_t L, PowSpec& wcl, size_t lminCMB, size_t lmaxCMB){
	
	std::vector< std::vector<double> > Fa, Fb;
	
	Fa=std::vector< std::vector<double> >(lmaxCMB+1, std::vector<double>(lmaxCMB+1,0.0));
	Fb=std::vector< std::vector<double> >(lmaxCMB+1, std::vector<double>(lmaxCMB+1,0.0));
	
	compF_pa(Fa, L, lmaxCMB+1);
	compF_pb(Fb, L, lmaxCMB+1);
	
	#pragma omp parallel for
	for (size_t l1=lminCMB;l1<lmaxCMB+1;l1++) {
		for (size_t l3=lminCMB;l3<lmaxCMB+1;l3++) {
			if ((l1+L+l3)%2==0) {
				f[0][l1][l3]=0;
				f[1][l1][l3]=0;
				f[2][l1][l3]=.5*wcl.tg(l1)*(Fb[l3][l1]-Fa[l3][l1]);
				f[3][l1][l3]=.5*wcl.gg(l1)*(Fb[l3][l1]-Fa[l3][l1])-.5*wcl.cc(l3)*(Fb[l1][l3]-Fa[l1][l3]);
			}
			else {
				f[0][l1][l3]=-.5*wcl.tg(l1)*(Fa[l3][l1]+Fb[l3][l1]);
				f[1][l1][l3]=-.5*wcl.gg(l1)*(Fa[l3][l1]+Fb[l3][l1])+.5*wcl.gg(l3)*(Fa[l1][l3]+Fb[l1][l3]);
				f[2][l1][l3]=0;
				f[3][l1][l3]=0;
			}
		}
	}
}

void computef_systda(std::vector< std::vector< std::vector<xcomplex< double >> > >& f, size_t L, PowSpec& wcl, size_t lminCMB, size_t lmaxCMB){
	
	std::vector< std::vector<double> > F;
	
	F=std::vector< std::vector<double> >(lmaxCMB+1, std::vector<double>(lmaxCMB+1,0.0));
	
	compF_d(F, L, lmaxCMB+1);
	
	#pragma omp parallel for
	for (size_t l1=lminCMB;l1<lmaxCMB+1;l1++) {
		for (size_t l3=lminCMB;l3<lmaxCMB+1;l3++) {
			if ((l1+L+l3)%2==0) {
				f[0][l1][l3]=-wcl.tt(l1)*F[l3][l1];
				f[1][l1][l3]=-wcl.tg(l1)*F[l3][l1]-wcl.tg(l3)*F[l1][l3];
				f[2][l1][l3]=0.0; 
				f[3][l1][l3]=0.0;
			}
			else {
				f[0][l1][l3]=0.0; 
				f[1][l1][l3]=0.0; 
				f[2][l1][l3]=+wcl.tt(l1)*F[l3][l1];
				f[3][l1][l3]=+wcl.tg(l1)*F[l3][l1];
			}
		}
	}
}

void computef_systdb(std::vector< std::vector< std::vector<xcomplex< double >> > >& f, size_t L, PowSpec& wcl, size_t lminCMB, size_t lmaxCMB){
	
	std::vector< std::vector<double> > F;
	
	F=std::vector< std::vector<double> >(lmaxCMB+1, std::vector<double>(lmaxCMB+1,0.0));
	
	compF_d(F, L, lmaxCMB+1);
	
	#pragma omp parallel for
	for (size_t l1=lminCMB;l1<lmaxCMB+1;l1++) {
		for (size_t l3=lminCMB;l3<lmaxCMB+1;l3++) {
			if ((l1+L+l3)%2!=0) {
				f[0][l1][l3]=+wcl.tt(l1)*F[l3][l1];
				f[1][l1][l3]=+wcl.tg(l1)*F[l3][l1]-wcl.tg(l3)*F[l1][l3];
				f[2][l1][l3]=0.0; 
				f[3][l1][l3]=0.0;
			}
			else {
				f[0][l1][l3]=0.0; 
				f[1][l1][l3]=0.0; 
				f[2][l1][l3]=-wcl.tt(l1)*F[l3][l1];
				f[3][l1][l3]=-wcl.tg(l1)*F[l3][l1];
			}
		}
	}
}

void computef_systq(std::vector< std::vector< std::vector<xcomplex< double >> > >& f, size_t L, PowSpec& wcl, size_t lminCMB, size_t lmaxCMB){
	
	std::vector< std::vector<double> > F;
	
	F=std::vector< std::vector<double> >(lmaxCMB+1, std::vector<double>(lmaxCMB+1,0.0));
	
	compF_q(F, L, lmaxCMB+1);
	
	#pragma omp parallel for
	for (size_t l1=lminCMB;l1<lmaxCMB+1;l1++) {
		for (size_t l3=lminCMB;l3<lmaxCMB+1;l3++) {
			if ((l1+L+l3)%2!=0) {
				f[0][l1][l3]=-wcl.tt(l1)*F[l3][l1];
				f[1][l1][l3]=-wcl.tg(l1)*F[l3][l1]+wcl.tg(l3)*F[l1][l3];
				f[2][l1][l3]=0.0; 
				f[3][l1][l3]=0.0;
			}
			else {
				f[0][l1][l3]=0.0; 
				f[1][l1][l3]=0.0; 
				f[2][l1][l3]=wcl.tt(l1)*F[l3][l1];
				f[3][l1][l3]=wcl.tg(l1)*F[l3][l1];
			}
		}
	}
}


std::vector< std::vector< std::vector< std::vector<double> > > > makeAN_syst(PowSpec& wcl, PowSpec& dcl, size_t lmin, size_t lmax, size_t lminCMB1, size_t lminCMB2, size_t lmaxCMB1, size_t lmaxCMB2) {
	int num_spec=4;
	int num_syst=11;

	assert(wcl.Num_specs()==4);
	
	size_t lmaxCMB=max(lmaxCMB1,lmaxCMB2);
	size_t lminCMB=min(lminCMB1,lminCMB2);

	std::vector< std::vector< std::vector< std::vector<double> > > > bias(lmax+1, std::vector<std::vector<std::vector<double>>>(num_syst ,std::vector<std::vector<double>>(num_syst,  std::vector<double>(num_spec,0.0))));
	std::vector< std::vector< std::vector< std::vector<xcomplex< double >> > > > f(num_syst,  std::vector< std::vector< std::vector<xcomplex< double >> > >(4, std::vector< std::vector<xcomplex< double >> >(lmaxCMB+1, std::vector<xcomplex< double >>(lmaxCMB+1,0.0))));

	
	std::vector<double> invlcltt(lmaxCMB+1,0.0), invlclee(lmaxCMB+1,0.0), invlclbb(lmaxCMB+1,0.0);
	
	#pragma omp parallel for
	for (size_t l1=lminCMB;l1<lmaxCMB+1;l1++) {
		invlcltt[l1]=1./dcl.tt(l1);
		invlclee[l1]=1./dcl.gg(l1);
		invlclbb[l1]=1./dcl.cc(l1);
	}
	
	double ate, aee, atb, aeb;
		
	
	for (size_t L=lmin;L<lmax+1;L++) {
		
		computef_systa(f[syst_a],L,wcl,lminCMB,lmaxCMB);
		computef_systomega(f[syst_o],L,wcl,lminCMB,lmaxCMB);
		computef_systfa(f[syst_f1],L,wcl,lminCMB,lmaxCMB);
		computef_systfb(f[syst_f2],L,wcl,lminCMB,lmaxCMB);
		computef_systgammaa(f[syst_g1],L,wcl,lminCMB,lmaxCMB);
		computef_systgammab(f[syst_g2],L,wcl,lminCMB,lmaxCMB);
		computef_systpa(f[syst_p1],L,wcl,lminCMB,lmaxCMB);
		computef_systpb(f[syst_p2],L,wcl,lminCMB,lmaxCMB);
		computef_systda(f[syst_d1],L,wcl,lminCMB,lmaxCMB);
		computef_systdb(f[syst_d2],L,wcl,lminCMB,lmaxCMB);
		computef_systq(f[syst_q],L,wcl,lminCMB,lmaxCMB);
				
		if (L>=lmin) {
			for (int s1=0;s1<num_syst;s1++) {
				for (int s2=0;s2<s1+1;s2++) {
					ate=0.; aee=0.; atb=0.; aeb=0.;
					#pragma omp parallel for reduction(+:ate, aee, atb, aeb) schedule(dynamic, 25)
					for (size_t l1=lminCMB1;l1<lmaxCMB1+1;l1++) {
						for (size_t l3=lminCMB2;l3<lmaxCMB2+1;l3++) {
							ate+=real(f[s1][0][l1][l3]*conj(f[s2][0][l1][l3]))*invlcltt[l1]*invlclee[l3];
							aee+=real(f[s1][1][l1][l3]*conj(f[s2][1][l1][l3]))*invlclee[l1]*invlclee[l3]*.5;
							atb+=real(f[s1][2][l1][l3]*conj(f[s2][2][l1][l3]))*invlcltt[l1]*invlclbb[l3]; 
							aeb+=real(f[s1][3][l1][l3]*conj(f[s2][3][l1][l3]))*invlclee[l1]*invlclbb[l3];
						}
					}
				
					bias[L][s1][s2][0] = (ate!=0.) ? (2.0*L+1.0)/ate : 0.0;
					bias[L][s1][s2][1] = (aee!=0.) ? (2.0*L+1.0)/aee : 0.0;
					bias[L][s1][s2][2] = (atb!=0.) ? (2.0*L+1.0)/atb : 0.0;
					bias[L][s1][s2][3] = (aeb!=0.) ? (2.0*L+1.0)/aeb : 0.0;
				}
			}
		}

		PyErr_CheckSignals();
	}
	
	return bias;
}


std::vector< std::vector<double> > makeA_syst(PowSpec& wcl, PowSpec& dcl, PowSpec& rdcls, PowSpec& al, size_t lmin, size_t lmax, size_t lminCMB, int type) {
	size_t lmaxCMB=dcl.Lmax();
			
	std::vector< std::vector< std::vector<xcomplex< double >> > > f(4, std::vector< std::vector<xcomplex< double >> >(lmaxCMB+1, std::vector<xcomplex< double >>(lmaxCMB+1,0.0)));

	std::vector< std::vector<double> > bias(6, std::vector<double>(lmax+1,0.0));
	std::vector<double> invlcltt(lmaxCMB+1,0.0), invlclee(lmaxCMB+1,0.0), invlclbb(lmaxCMB+1,0.0);
	
	#pragma omp parallel for
	for (size_t l1=lminCMB;l1<lmaxCMB+1;l1++) {
		invlcltt[l1]=1./dcl.tt(l1);
		invlclee[l1]=1./dcl.gg(l1);
		invlclbb[l1]=1./dcl.cc(l1);
	}
	
	double ate, aee, atb, aeb;
	double ntete, nteee, ntetb, nteeb, neeee, neetb, neeeb, ntbtb, ntbeb, nebeb;
	
	for (size_t L=lmin;L<lmax+1;L++) {
		
		// std::cout << " Computing amplitude ... " << (int)(L*100./lmax) << " %\r"; std::cout.flush();
		if      (type==0)  computef_systa(f,L,wcl,lminCMB,lmaxCMB);
		else if (type==1)  computef_systomega(f,L,wcl,lminCMB,lmaxCMB);
		else if (type==2) {computef_systfa(f,L,wcl,lminCMB,lmaxCMB);lmin=max((int)lmin,4);}
		else if (type==3) {computef_systfb(f,L,wcl,lminCMB,lmaxCMB);lmin=max((int)lmin,4);}
		else if (type==4)  computef_systgammaa(f,L,wcl,lminCMB,lmaxCMB);
		else if (type==5)  computef_systgammab(f,L,wcl,lminCMB,lmaxCMB);
		else if (type==6)  computef_systpa(f,L,wcl,lminCMB,lmaxCMB);
		else if (type==7)  computef_systpb(f,L,wcl,lminCMB,lmaxCMB);
		else if (type==8)  computef_systda(f,L,wcl,lminCMB,lmaxCMB);
		else if (type==9)  computef_systdb(f,L,wcl,lminCMB,lmaxCMB);
		else if (type==10) computef_systq(f,L,wcl,lminCMB,lmaxCMB);
		
		ate=0.; aee=0.; atb=0.; aeb=0.;
		ntete=0.; nteee=0.; ntetb=0.; nteeb=0.; neeee=0.; neetb=0.; neeeb=0.; ntbtb=0.; ntbeb=0.; nebeb=0.;
		
		if (L>=lmin) {
			#pragma omp parallel for reduction(+:ate, aee, atb, aeb, ntete, nteee, ntetb, nteeb, neeee, neetb, neeeb, ntbtb, ntbeb, nebeb) schedule(dynamic, 25)
			for (size_t l1=lminCMB;l1<lmaxCMB+1;l1++) {
				for (size_t l3=lminCMB;l3<lmaxCMB+1;l3++) {
					ate+=real(f[0][l1][l3]*conj(f[0][l1][l3]))*invlcltt[l1]*invlclee[l3];
					aee+=real(f[1][l1][l3]*conj(f[1][l1][l3]))*invlclee[l1]*invlclee[l3]*.5;
					atb+=real(f[2][l1][l3]*conj(f[2][l1][l3]))*invlcltt[l1]*invlclbb[l3]; 
					aeb+=real(f[3][l1][l3]*conj(f[3][l1][l3]))*invlclee[l1]*invlclbb[l3];

					ntete+=real(f[0][l1][l3]*invlcltt[l1]*invlclee[l3]*(conj(f[0][l1][l3])*invlcltt[l1]*invlclee[l3]*rdcls.tt(l1)*rdcls.gg(l3)+sgn(L+l1+l3)*conj(f[0][l3][l1])*invlcltt[l3]*invlclee[l1]*rdcls.tg(l1)*rdcls.tg(l3)));
					nteee+=real(f[0][l1][l3]*invlcltt[l1]*invlclee[l3]*(conj(f[1][l1][l3])*invlclee[l1]*invlclee[l3]*rdcls.tg(l1)*rdcls.gg(l3)+sgn(L+l1+l3)*conj(f[1][l3][l1])*invlclee[l3]*invlclee[l1]*rdcls.tg(l1)*rdcls.gg(l3))*.5);
					neeee+=real(f[1][l1][l3]*invlclee[l1]*invlclee[l3]*(conj(f[1][l1][l3])*invlclee[l1]*invlclee[l3]*rdcls.gg(l1)*rdcls.gg(l3)+sgn(L+l1+l3)*conj(f[1][l3][l1])*invlclee[l3]*invlclee[l1]*rdcls.gg(l1)*rdcls.gg(l3))*.25);
					ntbtb+=real(f[2][l1][l3]*invlcltt[l1]*invlclbb[l3]*(conj(f[2][l1][l3])*invlcltt[l1]*invlclbb[l3]*rdcls.tt(l1)*rdcls.cc(l3)));
					ntbeb+=real(f[2][l1][l3]*invlcltt[l1]*invlclbb[l3]*(conj(f[3][l1][l3])*invlclee[l1]*invlclbb[l3]*rdcls.tg(l1)*rdcls.cc(l3)));
					nebeb+=real(f[3][l1][l3]*invlclee[l1]*invlclbb[l3]*(conj(f[3][l1][l3])*invlclee[l1]*invlclbb[l3]*rdcls.gg(l1)*rdcls.cc(l3)));
				}
			}
		}
		
		al.tg(L) = (ate!=0.) ? (2.0*L+1.0)/ate : 0.0;
		al.gg(L) = (aee!=0.) ? (2.0*L+1.0)/aee : 0.0;
		al.tc(L) = (atb!=0.) ? (2.0*L+1.0)/atb : 0.0;
		al.gc(L) = (aeb!=0.) ? (2.0*L+1.0)/aeb : 0.0;
		
		bias[0][L]=ntete*al.tg(L)*al.tg(L)/(2.*L+1.);
		bias[1][L]=nteee*al.tg(L)*al.gg(L)/(2.*L+1.);
		bias[2][L]=neeee*al.gg(L)*al.gg(L)/(2.*L+1.);
		bias[3][L]=ntbtb*al.tc(L)*al.tc(L)/(2.*L+1.);
		bias[4][L]=ntbeb*al.tc(L)*al.gc(L)/(2.*L+1.);
		bias[5][L]=nebeb*al.gc(L)*al.gc(L)/(2.*L+1.);
		
		if(PyErr_CheckSignals() == -1) {
			throw invalid_argument( "Keyboard interrupt" );
		}
	}
	
	return bias;
}



void computef_dust(std::vector< std::vector< std::vector<double> > >& f, PowSpec& R1, PowSpec& R2, size_t L, PowSpec& wcl, size_t lminCMB, size_t lmaxCMB, int num_spec){
	std::vector< std::vector<double> > F;
	
	if (num_spec==6) {
		F=std::vector< std::vector<double> >(lmaxCMB+1, std::vector<double>(lmaxCMB+1,0.0));
		compF_2_mask(F, L, 2, lmaxCMB+1);
	}
	
	std::vector< std::vector<double> > Fz(lmaxCMB+1, std::vector<double>(lmaxCMB+1,0.0));
	compF_2_mask(Fz, L, 0, lmaxCMB+1);
	
	if (num_spec==1) {
		#pragma omp parallel for
		for (size_t l1=lminCMB;l1<lmaxCMB+1;l1++) {
			for (size_t l3=lminCMB;l3<lmaxCMB+1;l3++) {
				if ((l1+L+l3)%2==0) {
					f[tt][l1][l3]=wcl.tt(l1)*R2.tt(l3)*Fz[l3][l1]+wcl.tt(l3)*R1.tt(l1)*Fz[l1][l3];
				}
				else {
					f[tt][l1][l3]=0.0;
				}
			}
		}
	}
	else if (num_spec==6) {
		#pragma omp parallel for
		for (size_t l1=lminCMB;l1<lmaxCMB+1;l1++) {
			for (size_t l3=lminCMB;l3<lmaxCMB+1;l3++) {
				if ((l1+L+l3)%2==0) {
					f[tt][l1][l3]=wcl.tt(l1)*R2.tt(l3)*Fz[l3][l1]+wcl.tt(l3)*R1.tt(l1)*Fz[l1][l3];
					f[te][l1][l3]=wcl.tg(l1)*R2.gg(l3)*F[l3][l1]+wcl.tg(l3)*R1.tt(l1)*Fz[l1][l3];
					f[ee][l1][l3]=wcl.gg(l1)*R2.gg(l3)*F[l3][l1]+wcl.gg(l3)*R1.gg(l1)*F[l1][l3];
					f[tb][l1][l3]=0.0; 
					f[eb][l1][l3]=0.0;
					f[bb][l1][l3]=wcl.cc(l1)*R2.cc(l3)*F[l3][l1]+wcl.cc(l3)*R1.cc(l1)*F[l1][l3];
				}
				else {
					f[tt][l1][l3]=0.0;
					f[te][l1][l3]=0.0; 
					f[ee][l1][l3]=0.0; 
					f[tb][l1][l3]=-wcl.tg(l1)*R2.cc(l3)*F[l3][l1];
					f[eb][l1][l3]=-wcl.gg(l1)*R2.cc(l3)*F[l3][l1]+wcl.cc(l3)*R1.gg(l1)*F[l1][l3];
					f[bb][l1][l3]=0.0; 
				}
			}
		}
	}
	else std::cout << "I don't know what to do, yet" << std::endl;
}

std::vector< std::vector<double> > makeA_dust(PowSpec& wcl, PowSpec& dcl1, PowSpec& dcl2, PowSpec& R1, PowSpec& R2, PowSpec& rdcls, PowSpec& al, size_t lmin, size_t lmax, size_t lminCMB) {
	int num_spec=6;

	size_t lmaxCMB=dcl1.Lmax();

	assert(wcl.Num_specs()==4);


	std::vector< std::vector<double> > bias(10, std::vector<double>(lmax+1,0.0));
	std::vector< std::vector< std::vector<double> > > f(num_spec, std::vector< std::vector<double> >(lmaxCMB+1, std::vector<double>(lmaxCMB+1,0.0)));

	std::vector<double> invlcltt1(lmaxCMB+1,0.0), invlclee1(lmaxCMB+1,0.0), invlclbb1(lmaxCMB+1,0.0);
	std::vector<double> invlcltt2(lmaxCMB+1,0.0), invlclee2(lmaxCMB+1,0.0), invlclbb2(lmaxCMB+1,0.0);
	
	#pragma omp parallel for
	for (size_t l1=lminCMB;l1<lmaxCMB+1;l1++) {
		invlcltt1[l1]=1./dcl1.tt(l1);
		invlclee1[l1]=1./dcl1.gg(l1);
		invlclbb1[l1]=1./dcl1.cc(l1);
		invlcltt2[l1]=1./dcl2.tt(l1);
		invlclee2[l1]=1./dcl2.gg(l1);
		invlclbb2[l1]=1./dcl2.cc(l1);
	}
	
	double ntttt, nttte, nttee, ntete, nteee, neeee, ntbtb, ntbeb, nebeb, nbbbb, att, ate, aee, atb, aeb, abb;
	
	for (size_t L=lmin;L<lmax+1;L++) {
			
		computef_dust(f,R1,R2,L,wcl,lminCMB,lmaxCMB,num_spec);
		ntttt=0.; nttte=0.; nttee=0.; ntete=0.; nteee=0.; neeee=0.; ntbtb=0.; ntbeb=0.; nebeb=0.; nbbbb=0.;
		att=0.; ate=0.; aee=0.; atb=0.; aeb=0.; abb=0.;
		if (L>=lmin) {
			#pragma omp parallel for reduction(+:att, ate, aee, atb, aeb, ntttt, nttte, nttee, ntete, nteee, neeee, ntbtb, ntbeb, nebeb, nbbbb) schedule(dynamic, 25)
			for (size_t l1=lminCMB;l1<lmaxCMB+1;l1++) {
				for (size_t l3=lminCMB;l3<lmaxCMB+1;l3++) {
					att+=f[tt][l1][l3]*f[tt][l1][l3]*invlcltt1[l1]*invlcltt2[l3]*.5;
					ate+=f[te][l1][l3]*f[te][l1][l3]*invlcltt1[l1]*invlclee2[l3];
					aee+=f[ee][l1][l3]*f[ee][l1][l3]*invlclee1[l1]*invlclee2[l3]*.5;
					atb+=f[tb][l1][l3]*f[tb][l1][l3]*invlcltt1[l1]*invlclbb2[l3]; 
					aeb+=f[eb][l1][l3]*f[eb][l1][l3]*invlclee1[l1]*invlclbb2[l3];
					abb+=f[bb][l1][l3]*f[bb][l1][l3]*invlclbb1[l1]*invlclbb2[l3];

					ntttt+=f[tt][l1][l3]*invlcltt1[l1]*invlcltt2[l3]*(f[tt][l1][l3]*invlcltt1[l1]*invlcltt2[l3]*rdcls.tt(l1)*rdcls.tt(l3)+sgn(L+l1+l3)*f[tt][l3][l1]*invlcltt2[l3]*invlcltt1[l1]*rdcls.tt(l1)*rdcls.tt(l3))*.25;
					nttte+=f[tt][l1][l3]*invlcltt1[l1]*invlcltt2[l3]*(f[te][l1][l3]*invlcltt1[l1]*invlclee2[l3]*rdcls.tt(l1)*rdcls.tg(l3)+sgn(L+l1+l3)*f[te][l3][l1]*invlcltt2[l3]*invlclee1[l1]*rdcls.tg(l1)*rdcls.tt(l3))*.5;
					nttee+=f[tt][l1][l3]*invlcltt1[l1]*invlcltt2[l3]*(f[ee][l1][l3]*invlclee1[l1]*invlclee2[l3]*rdcls.tg(l1)*rdcls.tg(l3)+sgn(L+l1+l3)*f[ee][l3][l1]*invlclee2[l3]*invlclee1[l1]*rdcls.tg(l1)*rdcls.tg(l3))*.25;
					ntete+=f[te][l1][l3]*invlcltt1[l1]*invlclee2[l3]*(f[te][l1][l3]*invlcltt1[l1]*invlclee2[l3]*rdcls.tt(l1)*rdcls.gg(l3)+sgn(L+l1+l3)*f[te][l3][l1]*invlcltt2[l3]*invlclee1[l1]*rdcls.tg(l1)*rdcls.tg(l3));
					nteee+=f[te][l1][l3]*invlcltt1[l1]*invlclee2[l3]*(f[ee][l1][l3]*invlclee1[l1]*invlclee2[l3]*rdcls.tg(l1)*rdcls.gg(l3)+sgn(L+l1+l3)*f[ee][l3][l1]*invlclee2[l3]*invlclee1[l1]*rdcls.tg(l1)*rdcls.gg(l3))*.5;
					neeee+=f[ee][l1][l3]*invlclee1[l1]*invlclee2[l3]*(f[ee][l1][l3]*invlclee1[l1]*invlclee2[l3]*rdcls.gg(l1)*rdcls.gg(l3)+sgn(L+l1+l3)*f[ee][l3][l1]*invlclee2[l3]*invlclee1[l1]*rdcls.gg(l1)*rdcls.gg(l3))*.25;
					nbbbb+=f[bb][l1][l3]*invlclbb1[l1]*invlclbb2[l3]*(f[bb][l1][l3]*invlclbb1[l1]*invlclbb2[l3]*rdcls.cc(l1)*rdcls.cc(l3)+sgn(L+l1+l3)*f[bb][l3][l1]*invlclbb2[l3]*invlclbb1[l1]*rdcls.cc(l1)*rdcls.cc(l3))*.25;
					ntbtb+=f[tb][l1][l3]*invlcltt1[l1]*invlclbb2[l3]*(f[tb][l1][l3]*invlcltt1[l1]*invlclbb2[l3]*rdcls.tt(l1)*rdcls.cc(l3));
					ntbeb+=f[tb][l1][l3]*invlcltt1[l1]*invlclbb2[l3]*(f[eb][l1][l3]*invlclee1[l1]*invlclbb2[l3]*rdcls.tg(l1)*rdcls.cc(l3));
					nebeb+=f[eb][l1][l3]*invlclee1[l1]*invlclbb2[l3]*(f[eb][l1][l3]*invlclee1[l1]*invlclbb2[l3]*rdcls.gg(l1)*rdcls.cc(l3));
				}
			}
		}
		
		al.tt(L) = (att!=0.) ? (2.0*L+1.0)/att : 0.0;
		al.tg(L) = (ate!=0.) ? (2.0*L+1.0)/ate : 0.0;
		al.gg(L) = (aee!=0.) ? (2.0*L+1.0)/aee : 0.0;
		al.cc(L) = (abb!=0.) ? (2.0*L+1.0)/abb : 0.0;
		al.tc(L) = (atb!=0.) ? (2.0*L+1.0)/atb : 0.0;
		al.gc(L) = (aeb!=0.) ? (2.0*L+1.0)/aeb : 0.0;

		bias[tttt][L]=ntttt*al.tt(L)*al.tt(L)/(2.*L+1.);
		bias[ttte][L]=nttte*al.tt(L)*al.tg(L)/(2.*L+1.);
		bias[ttee][L]=nttee*al.tt(L)*al.gg(L)/(2.*L+1.);
		//=ntttb*att*atb/(2.*L+1.);
		//=ntteb*att*aeb/(2.*L+1.);
		bias[tete][L]=ntete*al.tg(L)*al.tg(L)/(2.*L+1.);
		bias[teee][L]=nteee*al.tg(L)*al.gg(L)/(2.*L+1.);
		//=ntetb*ate*atb/(2.*L+1.);
		//=nteeb*ate*aeb/(2.*L+1.);
		bias[eeee][L]=neeee*al.gg(L)*al.gg(L)/(2.*L+1.);
		//=neetb*aee*atb/(2.*L+1.);
		//=neeeb*aee*aeb/(2.*L+1.);
		bias[tbtb][L]=ntbtb*al.tc(L)*al.tc(L)/(2.*L+1.);
		bias[tbeb][L]=ntbeb*al.tc(L)*al.gc(L)/(2.*L+1.);
		bias[ebeb][L]=nebeb*al.gc(L)*al.gc(L)/(2.*L+1.);
		bias[bbbb][L]=nbbbb*al.cc(L)*al.cc(L)/(2.*L+1.);

		PyErr_CheckSignals();
	}
	
	return bias;
}


void computef_dust(std::vector< std::vector< std::vector<double> > >& f, std::vector<std::vector<double>>& R1, std::vector<std::vector<double>>& R2, size_t L, PowSpec& wcl, size_t lminCMB, size_t lmaxCMB, int num_spec){
	std::vector< std::vector<double> > F;
	
	if (num_spec==6) {
		F=std::vector< std::vector<double> >(lmaxCMB+1, std::vector<double>(lmaxCMB+1,0.0));
		compF_2_mask(F, L, 2, lmaxCMB+1);
	}
	
	std::vector< std::vector<double> > Fz(lmaxCMB+1, std::vector<double>(lmaxCMB+1,0.0));
	compF_2_mask(Fz, L, 0, lmaxCMB+1);
	
	if (num_spec==1) {
		#pragma omp parallel for
		for (size_t l1=lminCMB;l1<lmaxCMB+1;l1++) {
			for (size_t l3=lminCMB;l3<lmaxCMB+1;l3++) {
				if ((l1+L+l3)%2==0) {
					f[tt][l1][l3]=wcl.tt(l1)*R2[0][l3]*Fz[l3][l1]+wcl.tt(l3)*R1[0][l1]*Fz[l1][l3];
				}
				else {
					f[tt][l1][l3]=0.0;
				}
			}
		}
	}
	else if (num_spec==6) {
		#pragma omp parallel for
		for (size_t l1=lminCMB;l1<lmaxCMB+1;l1++) {
			for (size_t l3=lminCMB;l3<lmaxCMB+1;l3++) {
				if ((l1+L+l3)%2==0) {
					f[tt][l1][l3]=wcl.tt(l1)*R2[0][l3]*Fz[l3][l1]+wcl.tt(l3)*R1[0][l1]*Fz[l1][l3];
					f[te][l1][l3]=wcl.tg(l1)*R2[1][l3]*F[l3][l1]+wcl.tg(l3)*R1[0][l1]*Fz[l1][l3];
					f[ee][l1][l3]=wcl.gg(l1)*R2[1][l3]*F[l3][l1]+wcl.gg(l3)*R1[1][l1]*F[l1][l3];
					f[tb][l1][l3]=0.0; 
					f[eb][l1][l3]=0.0;
					f[bb][l1][l3]=wcl.cc(l1)*R2[2][l3]*F[l3][l1]+wcl.cc(l3)*R1[2][l1]*F[l1][l3];
				}
				else {
					f[tt][l1][l3]=0.0;
					f[te][l1][l3]=0.0; 
					f[ee][l1][l3]=0.0; 
					f[tb][l1][l3]=-wcl.tg(l1)*R2[1][l3]*F[l3][l1];
					f[eb][l1][l3]=-wcl.gg(l1)*R2[1][l3]*F[l3][l1]+wcl.cc(l3)*R1[2][l1]*F[l1][l3];
					f[bb][l1][l3]=0.0; 
				}
			}
		}
	}
	else std::cout << "I don't know what to do, yet" << std::endl;
}

#pragma omp declare reduction(vec_double_plus : std::vector< double > : std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<double>())) initializer(omp_priv = omp_orig)
std::vector< std::vector< std::vector< std::vector< std::vector<std::vector< double > > > > > > makeX_dust(PowSpec &wcl, std::vector< std::vector< std::vector<double> > > & dcl, std::vector< std::vector< std::vector< std::vector<double> > > > & rd, std::vector<std::vector< std::vector< std::vector<double> > > >& al, std::vector<std::vector<std::vector<double>>>& rlnu, size_t lmin, size_t lmax, size_t lminCMB1, size_t lminCMB2, size_t lmaxCMB1, size_t lmaxCMB2) {
	int num_spec=6;
	enum fourpoint {xtttt, xttte, xttee, xtttb, xtteb, xtett, xtete, xteee, xtetb, xteeb, xeett, xeete, xeeee, xeetb, xeeeb, xtbtt, xtbte, xtbee, xtbtb, xtbeb, xebtt, xebte, xebee, xebtb, xebeb, xbbbb};
	enum twopoint {xtt, xee, xbb, xte, xeb, xtb};

	assert(wcl.Num_specs()==4);
	
	size_t lmaxCMB=max(lmaxCMB1,lmaxCMB2);
	size_t lminCMB=min(lminCMB1,lminCMB2);

	std::vector< std::vector< std::vector< std::vector< std::vector<std::vector< double > > > > > > bias(lmax+1,std::vector<std::vector<std::vector<std::vector<std::vector<double>>>>>(dcl.size(),std::vector<std::vector<std::vector<std::vector<double>>>>(dcl.size(),std::vector<std::vector<std::vector<double>>>(dcl.size(),std::vector<std::vector<double>>(dcl.size(), std::vector<double>(10,0.0))))));	
	std::vector< std::vector< std::vector<double> > > f(num_spec, std::vector< std::vector<double> >(lmaxCMB+1, std::vector<double>(lmaxCMB+1,0.0)));

			
	#pragma omp parallel for
	for (size_t i=0; i<dcl.size(); i++) {
		for (size_t l=lminCMB;l<lmaxCMB+1;l++) {
			dcl[i][xtt][l]=1./dcl[i][xtt][l];
			dcl[i][xee][l]=1./dcl[i][xee][l];
			dcl[i][xbb][l]=1./dcl[i][xbb][l];
		}
	}
	 
	std::vector< double > temp_(10,0.0);
		
	for (size_t L=lmin;L<lmax+1;L++) {

		
		for (size_t a=0;a<dcl.size();a++) {
			for (size_t b=0;b<dcl.size();b++) {
				
				computef_dust(f,rlnu[a],rlnu[b],L,wcl,lminCMB,lmaxCMB,num_spec);
				
				
				std::fill(temp_.begin(), temp_.end(), 0.);
				#pragma omp parallel for reduction(vec_double_plus: temp_) schedule(guided)
				for (size_t l1=lminCMB1;l1<lmaxCMB1+1;l1++) {
					for (size_t l3=lminCMB2;l3<lmaxCMB2+1;l3++) {
						temp_[tt]+=f[tt][l1][l3]*f[tt][l1][l3]*dcl[a][xtt][l1]*dcl[b][xtt][l3]*.5;
						temp_[te]+=f[te][l1][l3]*f[te][l1][l3]*dcl[a][xtt][l1]*dcl[b][xee][l3];
						// temp_[et]+=f[et][l1][l3]*f[et][l1][l3]*dcl[a][xee][l1]*dcl[b][xtt][l3];
						temp_[ee]+=f[ee][l1][l3]*f[ee][l1][l3]*dcl[a][xee][l1]*dcl[b][xee][l3]*.5;
						temp_[tb]+=f[tb][l1][l3]*f[tb][l1][l3]*dcl[a][xtt][l1]*dcl[b][xbb][l3]; 
						// temp_[bt]+=f[bt][l1][l3]*f[bt][l1][l3]*dcl[a][xbb][l1]*dcl[b][xtt][l3]; 
						temp_[eb]+=f[eb][l1][l3]*f[eb][l1][l3]*dcl[a][xee][l1]*dcl[b][xbb][l3];
						// temp_[be]+=f[be][l1][l3]*f[be][l1][l3]*dcl[a][xbb][l1]*dcl[b][xee][l3];
						temp_[bb]+=f[bb][l1][l3]*f[bb][l1][l3]*dcl[a][xbb][l1]*dcl[b][xbb][l3]*.5;
					}
				}
				
				for ( int s = tt; s <= bb; s++ ) {
					al[L][a][b][s] = (temp_[s]!=0.) ? (2.0*L+1.0)/temp_[s] : 0.0;
				}
				
				for (size_t c=0;c<dcl.size();c++) {
					for (size_t d=0;d<dcl.size();d++) {
						std::fill(temp_.begin(), temp_.end(), 0.);
							
						if (L>=lmin) {
							#pragma omp parallel for reduction(vec_double_plus: temp_) schedule(guided)
							for (size_t l1=lminCMB1;l1<lmaxCMB1+1;l1++) {
								for (size_t l3=lminCMB2;l3<lmaxCMB2+1;l3++) {
									// temp_[astttt]+=f[astt][l1][l3]*dcl[a][xtt][l1]*dcl[b][xtt][l3]*(f[astt][l1][l3]*dcl[c][xtt][l1]*dcl[d][xtt][l3]*rd[a][c][xtt][l1]*rd[b][d][xtt][l3]+sgn(L+l1+l3)*f[astt][l3][l1]*dcl[c][xtt][l3]*dcl[d][xtt][l1]*rd[a][d][xtt][l1]*rd[b][c][xtt][l3])*.25;
									// temp_[asttte]+=f[astt][l1][l3]*dcl[a][xtt][l1]*dcl[b][xtt][l3]*(f[aste][l1][l3]*dcl[c][xtt][l1]*dcl[d][xee][l3]*rd[a][c][xtt][l1]*rd[b][d][xte][l3]+sgn(L+l1+l3)*f[aste][l3][l1]*dcl[c][xtt][l3]*dcl[d][xee][l1]*rd[a][d][xte][l1]*rd[b][c][xtt][l3])*.5;
									// temp_[asttet]+=f[astt][l1][l3]*dcl[a][xtt][l1]*dcl[b][xtt][l3]*(f[aset][l1][l3]*dcl[c][xee][l1]*dcl[d][xtt][l3]*rd[a][c][xte][l1]*rd[b][d][xtt][l3]+sgn(L+l1+l3)*f[aset][l3][l1]*dcl[c][xee][l3]*dcl[d][xtt][l1]*rd[a][d][xtt][l1]*rd[b][c][xte][l3])*.5;
									// temp_[asttee]+=f[astt][l1][l3]*dcl[a][xtt][l1]*dcl[b][xtt][l3]*(f[asee][l1][l3]*dcl[c][xee][l1]*dcl[d][xee][l3]*rd[a][c][xte][l1]*rd[b][d][xte][l3]+sgn(L+l1+l3)*f[asee][l3][l1]*dcl[c][xee][l3]*dcl[d][xee][l1]*rd[a][d][xte][l1]*rd[b][c][xte][l3])*.25;
									
									// temp_[astett]+=f[aste][l1][l3]*dcl[a][xtt][l1]*dcl[b][xee][l3]*(f[astt][l1][l3]*dcl[c][xtt][l1]*dcl[d][xtt][l3]*rd[a][c][xtt][l1]*rd[b][d][xte][l3]+sgn(L+l1+l3)*f[astt][l3][l1]*dcl[c][xtt][l3]*dcl[d][xtt][l1]*rd[a][d][xte][l1]*rd[b][c][xtt][l3])*.5;
									// temp_[astete]+=f[aste][l1][l3]*dcl[a][xtt][l1]*dcl[b][xee][l3]*(f[aste][l1][l3]*dcl[c][xtt][l1]*dcl[d][xee][l3]*rd[a][c][xtt][l1]*rd[b][d][xee][l3]+sgn(L+l1+l3)*f[aste][l3][l1]*dcl[c][xtt][l3]*dcl[d][xee][l1]*rd[a][d][xte][l1]*rd[b][c][xte][l3]);
									// temp_[asteet]+=f[aste][l1][l3]*dcl[a][xtt][l1]*dcl[b][xee][l3]*(f[aset][l1][l3]*dcl[c][xee][l1]*dcl[d][xtt][l3]*rd[a][c][xte][l1]*rd[b][d][xte][l3]+sgn(L+l1+l3)*f[aset][l3][l1]*dcl[c][xee][l3]*dcl[d][xtt][l1]*rd[a][d][xtt][l1]*rd[b][c][xee][l3]);
									// temp_[asteee]+=f[aste][l1][l3]*dcl[a][xtt][l1]*dcl[b][xee][l3]*(f[asee][l1][l3]*dcl[c][xee][l1]*dcl[d][xee][l3]*rd[a][c][xte][l1]*rd[b][d][xee][l3]+sgn(L+l1+l3)*f[asee][l3][l1]*dcl[c][xee][l3]*dcl[d][xee][l1]*rd[a][d][xte][l1]*rd[b][c][xee][l3])*.5;
									
									// temp_[asettt]+=f[aset][l1][l3]*dcl[a][xee][l1]*dcl[b][xtt][l3]*(f[astt][l1][l3]*dcl[c][xtt][l1]*dcl[d][xtt][l3]*rd[a][c][xte][l1]*rd[b][d][xtt][l3]+sgn(L+l1+l3)*f[astt][l3][l1]*dcl[c][xtt][l3]*dcl[d][xtt][l1]*rd[a][d][xte][l1]*rd[b][c][xtt][l3])*.5;
									// temp_[asette]+=f[aset][l1][l3]*dcl[a][xee][l1]*dcl[b][xtt][l3]*(f[aste][l1][l3]*dcl[c][xtt][l1]*dcl[d][xee][l3]*rd[a][c][xte][l1]*rd[b][d][xte][l3]+sgn(L+l1+l3)*f[aste][l3][l1]*dcl[c][xtt][l3]*dcl[d][xee][l1]*rd[a][d][xee][l1]*rd[b][c][xtt][l3]);
									// temp_[asetet]+=f[aset][l1][l3]*dcl[a][xee][l1]*dcl[b][xtt][l3]*(f[aset][l1][l3]*dcl[c][xee][l1]*dcl[d][xtt][l3]*rd[a][c][xee][l1]*rd[b][d][xtt][l3]+sgn(L+l1+l3)*f[aset][l3][l1]*dcl[c][xee][l3]*dcl[d][xtt][l1]*rd[a][d][xte][l1]*rd[b][c][xte][l3]);
									// temp_[asetee]+=f[aset][l1][l3]*dcl[a][xee][l1]*dcl[b][xtt][l3]*(f[asee][l1][l3]*dcl[c][xee][l1]*dcl[d][xee][l3]*rd[a][c][xee][l1]*rd[b][d][xte][l3]+sgn(L+l1+l3)*f[asee][l3][l1]*dcl[c][xee][l3]*dcl[d][xee][l1]*rd[a][d][xee][l1]*rd[b][c][xte][l3])*.5;

									// temp_[aseett]+=f[asee][l1][l3]*dcl[a][xee][l1]*dcl[b][xee][l3]*(f[astt][l1][l3]*dcl[c][xtt][l1]*dcl[d][xtt][l3]*rd[a][c][xte][l1]*rd[b][d][xte][l3]+sgn(L+l1+l3)*f[astt][l3][l1]*dcl[c][xtt][l3]*dcl[d][xtt][l1]*rd[a][d][xte][l1]*rd[b][c][xte][l3])*.25;
									// temp_[aseete]+=f[asee][l1][l3]*dcl[a][xee][l1]*dcl[b][xee][l3]*(f[aste][l1][l3]*dcl[c][xtt][l1]*dcl[d][xee][l3]*rd[a][c][xte][l1]*rd[b][d][xee][l3]+sgn(L+l1+l3)*f[aste][l3][l1]*dcl[c][xtt][l3]*dcl[d][xee][l1]*rd[a][d][xee][l1]*rd[b][c][xte][l3])*.5;
									// temp_[aseeet]+=f[asee][l1][l3]*dcl[a][xee][l1]*dcl[b][xee][l3]*(f[aset][l1][l3]*dcl[c][xee][l1]*dcl[d][xtt][l3]*rd[a][c][xee][l1]*rd[b][d][xte][l3]+sgn(L+l1+l3)*f[aset][l3][l1]*dcl[c][xee][l3]*dcl[d][xtt][l1]*rd[a][d][xte][l1]*rd[b][c][xee][l3])*.5;
									// temp_[aseeee]+=f[asee][l1][l3]*dcl[a][xee][l1]*dcl[b][xee][l3]*(f[asee][l1][l3]*dcl[c][xee][l1]*dcl[d][xee][l3]*rd[a][c][xee][l1]*rd[b][d][xee][l3]+sgn(L+l1+l3)*f[asee][l3][l1]*dcl[c][xee][l3]*dcl[d][xee][l1]*rd[a][d][xee][l1]*rd[b][c][xee][l3])*.25;

									// temp_[astbtb]+=f[tb][l1][l3]*dcl[a][xtt][l1]*dcl[b][xbb][l3]*(f[tb][l1][l3]*dcl[c][xtt][l1]*dcl[d][xbb][l3]*rd[a][c][xtt][l1]*rd[b][d][xbb][l3]+sgn(L+l1+l3)*f[tb][l3][l1]*dcl[c][xtt][l3]*dcl[d][xbb][l1]*rd[a][d][xtb][l1]*rd[b][c][xtb][l3]);
									// temp_[astbbt]+=f[tb][l1][l3]*dcl[a][xtt][l1]*dcl[b][xbb][l3]*(f[bt][l1][l3]*dcl[c][xbb][l1]*dcl[d][xtt][l3]*rd[a][c][xtb][l1]*rd[b][d][xtb][l3]+sgn(L+l1+l3)*f[bt][l3][l1]*dcl[c][xbb][l3]*dcl[d][xtt][l1]*rd[a][d][xtt][l1]*rd[b][c][xbb][l3]);
									// temp_[astbeb]+=f[tb][l1][l3]*dcl[a][xtt][l1]*dcl[b][xbb][l3]*(f[eb][l1][l3]*dcl[c][xee][l1]*dcl[d][xbb][l3]*rd[a][c][xte][l1]*rd[b][d][xbb][l3]+sgn(L+l1+l3)*f[eb][l3][l1]*dcl[c][xee][l3]*dcl[d][xbb][l1]*rd[a][d][xtb][l1]*rd[b][c][xeb][l3]);
									// temp_[astbbe]+=f[tb][l1][l3]*dcl[a][xtt][l1]*dcl[b][xbb][l3]*(f[be][l1][l3]*dcl[c][xbb][l1]*dcl[d][xee][l3]*rd[a][c][xtb][l1]*rd[b][d][xeb][l3]+sgn(L+l1+l3)*f[be][l3][l1]*dcl[c][xbb][l3]*dcl[d][xee][l1]*rd[a][d][xte][l1]*rd[b][c][xbb][l3]);

									// temp_[asbttb]+=f[bt][l1][l3]*dcl[a][xbb][l1]*dcl[b][xtt][l3]*(f[tb][l1][l3]*dcl[c][xtt][l1]*dcl[d][xbb][l3]*rd[a][c][xtb][l1]*rd[b][d][xtb][l3]+sgn(L+l1+l3)*f[tb][l3][l1]*dcl[c][xtt][l3]*dcl[d][xbb][l1]*rd[a][d][xbb][l1]*rd[b][c][xtt][l3]);
									// temp_[asbtbt]+=f[bt][l1][l3]*dcl[a][xbb][l1]*dcl[b][xtt][l3]*(f[bt][l1][l3]*dcl[c][xbb][l1]*dcl[d][xtt][l3]*rd[a][c][xbb][l1]*rd[b][d][xtt][l3]+sgn(L+l1+l3)*f[bt][l3][l1]*dcl[c][xbb][l3]*dcl[d][xtt][l1]*rd[a][d][xtb][l1]*rd[b][c][xtb][l3]);
									// temp_[asbteb]+=f[bt][l1][l3]*dcl[a][xbb][l1]*dcl[b][xtt][l3]*(f[eb][l1][l3]*dcl[c][xee][l1]*dcl[d][xbb][l3]*rd[a][c][xeb][l1]*rd[b][d][xtb][l3]+sgn(L+l1+l3)*f[eb][l3][l1]*dcl[c][xee][l3]*dcl[d][xbb][l1]*rd[a][d][xbb][l1]*rd[b][c][xte][l3]);
									// temp_[asbtbe]+=f[bt][l1][l3]*dcl[a][xbb][l1]*dcl[b][xtt][l3]*(f[be][l1][l3]*dcl[c][xbb][l1]*dcl[d][xee][l3]*rd[a][c][xbb][l1]*rd[b][d][xte][l3]+sgn(L+l1+l3)*f[be][l3][l1]*dcl[c][xbb][l3]*dcl[d][xee][l1]*rd[a][d][xeb][l1]*rd[b][c][xtb][l3]);

									// temp_[asebtb]+=f[eb][l1][l3]*dcl[a][xee][l1]*dcl[b][xbb][l3]*(f[tb][l1][l3]*dcl[c][xtt][l1]*dcl[d][xbb][l3]*rd[a][c][xte][l1]*rd[b][d][xbb][l3]+sgn(L+l1+l3)*f[tb][l3][l1]*dcl[c][xtt][l3]*dcl[d][xbb][l1]*rd[a][d][xeb][l1]*rd[b][c][xtb][l3]);
									// temp_[asebbt]+=f[eb][l1][l3]*dcl[a][xee][l1]*dcl[b][xbb][l3]*(f[bt][l1][l3]*dcl[c][xbb][l1]*dcl[d][xtt][l3]*rd[a][c][xeb][l1]*rd[b][d][xtb][l3]+sgn(L+l1+l3)*f[bt][l3][l1]*dcl[c][xbb][l3]*dcl[d][xtt][l1]*rd[a][d][xte][l1]*rd[b][c][xbb][l3]);
									// temp_[asebeb]+=f[eb][l1][l3]*dcl[a][xee][l1]*dcl[b][xbb][l3]*(f[eb][l1][l3]*dcl[c][xee][l1]*dcl[d][xbb][l3]*rd[a][c][xee][l1]*rd[b][d][xbb][l3]+sgn(L+l1+l3)*f[eb][l3][l1]*dcl[c][xee][l3]*dcl[d][xbb][l1]*rd[a][d][xeb][l1]*rd[b][c][xeb][l3]);
									// temp_[asebbe]+=f[eb][l1][l3]*dcl[a][xee][l1]*dcl[b][xbb][l3]*(f[be][l1][l3]*dcl[c][xbb][l1]*dcl[d][xee][l3]*rd[a][c][xeb][l1]*rd[b][d][xeb][l3]+sgn(L+l1+l3)*f[be][l3][l1]*dcl[c][xbb][l3]*dcl[d][xee][l1]*rd[a][d][xee][l1]*rd[b][c][xbb][l3]);
				
									// temp_[asbetb]+=f[be][l1][l3]*dcl[a][xbb][l1]*dcl[b][xee][l3]*(f[tb][l1][l3]*dcl[c][xtt][l1]*dcl[d][xbb][l3]*rd[a][c][xtb][l1]*rd[b][d][xeb][l3]+sgn(L+l1+l3)*f[tb][l3][l1]*dcl[c][xtt][l3]*dcl[d][xbb][l1]*rd[a][d][xbb][l1]*rd[b][c][xte][l3]);
									// temp_[asbebt]+=f[be][l1][l3]*dcl[a][xbb][l1]*dcl[b][xee][l3]*(f[bt][l1][l3]*dcl[c][xbb][l1]*dcl[d][xtt][l3]*rd[a][c][xbb][l1]*rd[b][d][xte][l3]+sgn(L+l1+l3)*f[bt][l3][l1]*dcl[c][xbb][l3]*dcl[d][xtt][l1]*rd[a][d][xtb][l1]*rd[b][c][xeb][l3]);
									// temp_[asbeeb]+=f[be][l1][l3]*dcl[a][xbb][l1]*dcl[b][xee][l3]*(f[eb][l1][l3]*dcl[c][xee][l1]*dcl[d][xbb][l3]*rd[a][c][xeb][l1]*rd[b][d][xeb][l3]+sgn(L+l1+l3)*f[eb][l3][l1]*dcl[c][xee][l3]*dcl[d][xbb][l1]*rd[a][d][xbb][l1]*rd[b][c][xee][l3]);
									// temp_[asbebe]+=f[be][l1][l3]*dcl[a][xbb][l1]*dcl[b][xee][l3]*(f[be][l1][l3]*dcl[c][xbb][l1]*dcl[d][xee][l3]*rd[a][c][xbb][l1]*rd[b][d][xee][l3]+sgn(L+l1+l3)*f[be][l3][l1]*dcl[c][xbb][l3]*dcl[d][xee][l1]*rd[a][d][xeb][l1]*rd[b][c][xeb][l3]);
									
									// temp_[astttt]+=f[astt][l1][l3]*dcl[a][xtt][l1]*dcl[b][xtt][l3]*(f[astt][l1][l3]*dcl[c][xtt][l1]*dcl[d][xtt][l3]*rd[a][c][xtt][l1]*rd[b][d][xtt][l3]+sgn(L+l1+l3)*f[astt][l3][l1]*dcl[c][xtt][l3]*dcl[d][xtt][l1]*rd[a][d][xtt][l1]*rd[b][c][xtt][l3])*.25;
									// temp_[asttte]+=f[astt][l1][l3]*dcl[a][xtt][l1]*dcl[b][xtt][l3]*(f[aste][l1][l3]*dcl[c][xtt][l1]*dcl[d][xee][l3]*rd[a][c][xtt][l1]*rd[b][d][xte][l3]+sgn(L+l1+l3)*f[aste][l3][l1]*dcl[c][xtt][l3]*dcl[d][xee][l1]*rd[a][d][xte][l1]*rd[b][c][xtt][l3])*.5;
									// temp_[asttet]+=f[astt][l1][l3]*dcl[a][xtt][l1]*dcl[b][xtt][l3]*(f[aset][l1][l3]*dcl[c][xee][l1]*dcl[d][xtt][l3]*rd[a][c][xte][l1]*rd[b][d][xtt][l3]+sgn(L+l1+l3)*f[aset][l3][l1]*dcl[c][xee][l3]*dcl[d][xtt][l1]*rd[a][d][xtt][l1]*rd[b][c][xte][l3])*.5;
									// temp_[asttee]+=f[astt][l1][l3]*dcl[a][xtt][l1]*dcl[b][xtt][l3]*(f[asee][l1][l3]*dcl[c][xee][l1]*dcl[d][xee][l3]*rd[a][c][xte][l1]*rd[b][d][xte][l3]+sgn(L+l1+l3)*f[asee][l3][l1]*dcl[c][xee][l3]*dcl[d][xee][l1]*rd[a][d][xte][l1]*rd[b][c][xte][l3])*.25;
									
									// temp_[astett]+=f[aste][l1][l3]*dcl[a][xtt][l1]*dcl[b][xee][l3]*(f[astt][l1][l3]*dcl[c][xtt][l1]*dcl[d][xtt][l3]*rd[a][c][xtt][l1]*rd[b][d][xte][l3]+sgn(L+l1+l3)*f[astt][l3][l1]*dcl[c][xtt][l3]*dcl[d][xtt][l1]*rd[a][d][xte][l1]*rd[b][c][xtt][l3])*.5;
									// temp_[astete]+=f[aste][l1][l3]*dcl[a][xtt][l1]*dcl[b][xee][l3]*(f[aste][l1][l3]*dcl[c][xtt][l1]*dcl[d][xee][l3]*rd[a][c][xtt][l1]*rd[b][d][xee][l3]+sgn(L+l1+l3)*f[aste][l3][l1]*dcl[c][xtt][l3]*dcl[d][xee][l1]*rd[a][d][xte][l1]*rd[b][c][xte][l3]);
									// temp_[asteet]+=f[aste][l1][l3]*dcl[a][xtt][l1]*dcl[b][xee][l3]*(f[aset][l1][l3]*dcl[c][xee][l1]*dcl[d][xtt][l3]*rd[a][c][xte][l1]*rd[b][d][xte][l3]+sgn(L+l1+l3)*f[aset][l3][l1]*dcl[c][xee][l3]*dcl[d][xtt][l1]*rd[a][d][xtt][l1]*rd[b][c][xee][l3]);
									// temp_[asteee]+=f[aste][l1][l3]*dcl[a][xtt][l1]*dcl[b][xee][l3]*(f[asee][l1][l3]*dcl[c][xee][l1]*dcl[d][xee][l3]*rd[a][c][xte][l1]*rd[b][d][xee][l3]+sgn(L+l1+l3)*f[asee][l3][l1]*dcl[c][xee][l3]*dcl[d][xee][l1]*rd[a][d][xte][l1]*rd[b][c][xee][l3])*.5;
									
									// temp_[asettt]+=f[aset][l1][l3]*dcl[a][xee][l1]*dcl[b][xtt][l3]*(f[astt][l1][l3]*dcl[c][xtt][l1]*dcl[d][xtt][l3]*rd[a][c][xte][l1]*rd[b][d][xtt][l3]+sgn(L+l1+l3)*f[astt][l3][l1]*dcl[c][xtt][l3]*dcl[d][xtt][l1]*rd[a][d][xte][l1]*rd[b][c][xtt][l3])*.5;
									// temp_[asette]+=f[aset][l1][l3]*dcl[a][xee][l1]*dcl[b][xtt][l3]*(f[aste][l1][l3]*dcl[c][xtt][l1]*dcl[d][xee][l3]*rd[a][c][xte][l1]*rd[b][d][xte][l3]+sgn(L+l1+l3)*f[aste][l3][l1]*dcl[c][xtt][l3]*dcl[d][xee][l1]*rd[a][d][xee][l1]*rd[b][c][xtt][l3]);
									// temp_[asetet]+=f[aset][l1][l3]*dcl[a][xee][l1]*dcl[b][xtt][l3]*(f[aset][l1][l3]*dcl[c][xee][l1]*dcl[d][xtt][l3]*rd[a][c][xee][l1]*rd[b][d][xtt][l3]+sgn(L+l1+l3)*f[aset][l3][l1]*dcl[c][xee][l3]*dcl[d][xtt][l1]*rd[a][d][xte][l1]*rd[b][c][xte][l3]);
									// temp_[asetee]+=f[aset][l1][l3]*dcl[a][xee][l1]*dcl[b][xtt][l3]*(f[asee][l1][l3]*dcl[c][xee][l1]*dcl[d][xee][l3]*rd[a][c][xee][l1]*rd[b][d][xte][l3]+sgn(L+l1+l3)*f[asee][l3][l1]*dcl[c][xee][l3]*dcl[d][xee][l1]*rd[a][d][xee][l1]*rd[b][c][xte][l3])*.5;

									// temp_[aseett]+=f[asee][l1][l3]*dcl[a][xee][l1]*dcl[b][xee][l3]*(f[astt][l1][l3]*dcl[c][xtt][l1]*dcl[d][xtt][l3]*rd[a][c][xte][l1]*rd[b][d][xte][l3]+sgn(L+l1+l3)*f[astt][l3][l1]*dcl[c][xtt][l3]*dcl[d][xtt][l1]*rd[a][d][xte][l1]*rd[b][c][xte][l3])*.25;
									// temp_[aseete]+=f[asee][l1][l3]*dcl[a][xee][l1]*dcl[b][xee][l3]*(f[aste][l1][l3]*dcl[c][xtt][l1]*dcl[d][xee][l3]*rd[a][c][xte][l1]*rd[b][d][xee][l3]+sgn(L+l1+l3)*f[aste][l3][l1]*dcl[c][xtt][l3]*dcl[d][xee][l1]*rd[a][d][xee][l1]*rd[b][c][xte][l3])*.5;
									// temp_[aseeet]+=f[asee][l1][l3]*dcl[a][xee][l1]*dcl[b][xee][l3]*(f[aset][l1][l3]*dcl[c][xee][l1]*dcl[d][xtt][l3]*rd[a][c][xee][l1]*rd[b][d][xte][l3]+sgn(L+l1+l3)*f[aset][l3][l1]*dcl[c][xee][l3]*dcl[d][xtt][l1]*rd[a][d][xte][l1]*rd[b][c][xee][l3])*.5;
									// temp_[aseeee]+=f[asee][l1][l3]*dcl[a][xee][l1]*dcl[b][xee][l3]*(f[asee][l1][l3]*dcl[c][xee][l1]*dcl[d][xee][l3]*rd[a][c][xee][l1]*rd[b][d][xee][l3]+sgn(L+l1+l3)*f[asee][l3][l1]*dcl[c][xee][l3]*dcl[d][xee][l1]*rd[a][d][xee][l1]*rd[b][c][xee][l3])*.25;

									// temp_[astbtb]+=f[astb][l1][l3]*dcl[a][xtt][l1]*dcl[b][xbb][l3]*(f[astb][l1][l3]*dcl[c][xtt][l1]*dcl[d][xbb][l3]*rd[a][c][xtt][l1]*rd[b][d][xbb][l3]);
									// temp_[astbbt]+=f[astb][l1][l3]*dcl[a][xtt][l1]*dcl[b][xbb][l3]*(sgn(L+l1+l3)*f[asbt][l3][l1]*dcl[c][xbb][l3]*dcl[d][xtt][l1]*rd[a][d][xtt][l1]*rd[b][c][xbb][l3]);
									// temp_[astbeb]+=f[astb][l1][l3]*dcl[a][xtt][l1]*dcl[b][xbb][l3]*(f[aseb][l1][l3]*dcl[c][xee][l1]*dcl[d][xbb][l3]*rd[a][c][xte][l1]*rd[b][d][xbb][l3]);
									// temp_[astbbe]+=f[astb][l1][l3]*dcl[a][xtt][l1]*dcl[b][xbb][l3]*(sgn(L+l1+l3)*f[asbe][l3][l1]*dcl[c][xbb][l3]*dcl[d][xee][l1]*rd[a][d][xte][l1]*rd[b][c][xbb][l3]);

									// temp_[asbttb]+=f[asbt][l1][l3]*dcl[a][xbb][l1]*dcl[b][xtt][l3]*(sgn(L+l1+l3)*f[astb][l3][l1]*dcl[c][xtt][l3]*dcl[d][xbb][l1]*rd[a][d][xbb][l1]*rd[b][c][xtt][l3]);
									// temp_[asbtbt]+=f[asbt][l1][l3]*dcl[a][xbb][l1]*dcl[b][xtt][l3]*(f[asbt][l1][l3]*dcl[c][xbb][l1]*dcl[d][xtt][l3]*rd[a][c][xbb][l1]*rd[b][d][xtt][l3]);
									// temp_[asbteb]+=f[asbt][l1][l3]*dcl[a][xbb][l1]*dcl[b][xtt][l3]*(sgn(L+l1+l3)*f[aseb][l3][l1]*dcl[c][xee][l3]*dcl[d][xbb][l1]*rd[a][d][xbb][l1]*rd[b][c][xte][l3]);
									// temp_[asbtbe]+=f[asbt][l1][l3]*dcl[a][xbb][l1]*dcl[b][xtt][l3]*(f[asbe][l1][l3]*dcl[c][xbb][l1]*dcl[d][xee][l3]*rd[a][c][xbb][l1]*rd[b][d][xte][l3]);

									// temp_[asebtb]+=f[aseb][l1][l3]*dcl[a][xee][l1]*dcl[b][xbb][l3]*(f[astb][l1][l3]*dcl[c][xtt][l1]*dcl[d][xbb][l3]*rd[a][c][xte][l1]*rd[b][d][xbb][l3]);
									// temp_[asebbt]+=f[aseb][l1][l3]*dcl[a][xee][l1]*dcl[b][xbb][l3]*(sgn(L+l1+l3)*f[asbt][l3][l1]*dcl[c][xbb][l3]*dcl[d][xtt][l1]*rd[a][d][xte][l1]*rd[b][c][xbb][l3]);
									// temp_[asebeb]+=f[aseb][l1][l3]*dcl[a][xee][l1]*dcl[b][xbb][l3]*(f[aseb][l1][l3]*dcl[c][xee][l1]*dcl[d][xbb][l3]*rd[a][c][xee][l1]*rd[b][d][xbb][l3]);
									// temp_[asebbe]+=f[aseb][l1][l3]*dcl[a][xee][l1]*dcl[b][xbb][l3]*(sgn(L+l1+l3)*f[asbe][l3][l1]*dcl[c][xbb][l3]*dcl[d][xee][l1]*rd[a][d][xee][l1]*rd[b][c][xbb][l3]);
				
									// temp_[asbetb]+=f[asbe][l1][l3]*dcl[a][xbb][l1]*dcl[b][xee][l3]*(sgn(L+l1+l3)*f[astb][l3][l1]*dcl[c][xtt][l3]*dcl[d][xbb][l1]*rd[a][d][xbb][l1]*rd[b][c][xte][l3]);
									// temp_[asbebt]+=f[asbe][l1][l3]*dcl[a][xbb][l1]*dcl[b][xee][l3]*(f[asbt][l1][l3]*dcl[c][xbb][l1]*dcl[d][xtt][l3]*rd[a][c][xbb][l1]*rd[b][d][xte][l3]);
									// temp_[asbeeb]+=f[asbe][l1][l3]*dcl[a][xbb][l1]*dcl[b][xee][l3]*(sgn(L+l1+l3)*f[aseb][l3][l1]*dcl[c][xee][l3]*dcl[d][xbb][l1]*rd[a][d][xbb][l1]*rd[b][c][xee][l3]);
									// temp_[asbebe]+=f[asbe][l1][l3]*dcl[a][xbb][l1]*dcl[b][xee][l3]*(f[asbe][l1][l3]*dcl[c][xbb][l1]*dcl[d][xee][l3]*rd[a][c][xbb][l1]*rd[b][d][xee][l3]);
									
									
									temp_[tttt]+=f[tt][l1][l3]*dcl[a][xtt][l1]*dcl[b][xtt][l3]*(f[tt][l1][l3]*dcl[c][xtt][l1]*dcl[d][xtt][l3]*rd[a][c][xtt][l1]*rd[b][d][xtt][l3]+sgn(L+l1+l3)*f[tt][l3][l1]*dcl[c][xtt][l3]*dcl[d][xtt][l1]*rd[a][d][xtt][l1]*rd[b][c][xtt][l3])*.25;
									temp_[ttte]+=f[tt][l1][l3]*dcl[a][xtt][l1]*dcl[b][xtt][l3]*(f[te][l1][l3]*dcl[c][xtt][l1]*dcl[d][xee][l3]*rd[a][c][xtt][l1]*rd[b][d][xte][l3]+sgn(L+l1+l3)*f[te][l3][l1]*dcl[c][xtt][l3]*dcl[d][xee][l1]*rd[a][d][xte][l1]*rd[b][c][xtt][l3])*.5;
									temp_[ttee]+=f[tt][l1][l3]*dcl[a][xtt][l1]*dcl[b][xtt][l3]*(f[ee][l1][l3]*dcl[c][xee][l1]*dcl[d][xee][l3]*rd[a][c][xte][l1]*rd[b][d][xte][l3]+sgn(L+l1+l3)*f[ee][l3][l1]*dcl[c][xee][l3]*dcl[d][xee][l1]*rd[a][d][xte][l1]*rd[b][c][xte][l3])*.25;
									
									temp_[tete]+=f[te][l1][l3]*dcl[a][xtt][l1]*dcl[b][xee][l3]*(f[te][l1][l3]*dcl[c][xtt][l1]*dcl[d][xee][l3]*rd[a][c][xtt][l1]*rd[b][d][xee][l3]+sgn(L+l1+l3)*f[te][l3][l1]*dcl[c][xtt][l3]*dcl[d][xee][l1]*rd[a][d][xte][l1]*rd[b][c][xte][l3]);
									temp_[teee]+=f[te][l1][l3]*dcl[a][xtt][l1]*dcl[b][xee][l3]*(f[ee][l1][l3]*dcl[c][xee][l1]*dcl[d][xee][l3]*rd[a][c][xte][l1]*rd[b][d][xee][l3]+sgn(L+l1+l3)*f[ee][l3][l1]*dcl[c][xee][l3]*dcl[d][xee][l1]*rd[a][d][xte][l1]*rd[b][c][xee][l3])*.5;

									temp_[eeee]+=f[ee][l1][l3]*dcl[a][xee][l1]*dcl[b][xee][l3]*(f[ee][l1][l3]*dcl[c][xee][l1]*dcl[d][xee][l3]*rd[a][c][xee][l1]*rd[b][d][xee][l3]+sgn(L+l1+l3)*f[ee][l3][l1]*dcl[c][xee][l3]*dcl[d][xee][l1]*rd[a][d][xee][l1]*rd[b][c][xee][l3])*.25;

									temp_[tbtb]+=f[tb][l1][l3]*dcl[a][xtt][l1]*dcl[b][xbb][l3]*(f[tb][l1][l3]*dcl[c][xtt][l1]*dcl[d][xbb][l3]*rd[a][c][xtt][l1]*rd[b][d][xbb][l3]);
									temp_[tbeb]+=f[tb][l1][l3]*dcl[a][xtt][l1]*dcl[b][xbb][l3]*(f[eb][l1][l3]*dcl[c][xee][l1]*dcl[d][xbb][l3]*rd[a][c][xte][l1]*rd[b][d][xbb][l3]);

									temp_[ebeb]+=f[eb][l1][l3]*dcl[a][xee][l1]*dcl[b][xbb][l3]*(f[eb][l1][l3]*dcl[c][xee][l1]*dcl[d][xbb][l3]*rd[a][c][xee][l1]*rd[b][d][xbb][l3]);
									
									temp_[bbbb]+=f[bb][l1][l3]*dcl[a][xbb][l1]*dcl[b][xbb][l3]*(f[bb][l1][l3]*dcl[c][xbb][l1]*dcl[d][xbb][l3]*rd[a][c][xbb][l1]*rd[b][d][xbb][l3]+sgn(L+l1+l3)*f[bb][l3][l1]*dcl[c][xbb][l3]*dcl[d][xbb][l1]*rd[a][d][xbb][l1]*rd[b][c][xbb][l3])*.25;
								}
							}
						}

						// bias[L][a][b][c][d][astttt]=temp_[astttt]*al[L][a][b][astt]*al[L][c][d][astt]/(2.*L+1.);
						// bias[L][a][b][c][d][asttte]=temp_[asttte]*al[L][a][b][astt]*al[L][c][d][aste]/(2.*L+1.);
						// bias[L][a][b][c][d][asttet]=temp_[asttet]*al[L][a][b][astt]*al[L][c][d][aset]/(2.*L+1.);
						// bias[L][a][b][c][d][asttee]=temp_[asttee]*al[L][a][b][astt]*al[L][c][d][asee]/(2.*L+1.);
						// bias[L][a][b][c][d][astett]=temp_[astett]*al[L][a][b][aste]*al[L][c][d][astt]/(2.*L+1.);
						// bias[L][a][b][c][d][astete]=temp_[astete]*al[L][a][b][aste]*al[L][c][d][aste]/(2.*L+1.);
						// bias[L][a][b][c][d][asteet]=temp_[asteet]*al[L][a][b][aste]*al[L][c][d][aset]/(2.*L+1.);
						// bias[L][a][b][c][d][asteee]=temp_[asteee]*al[L][a][b][aste]*al[L][c][d][asee]/(2.*L+1.);
						// bias[L][a][b][c][d][asettt]=temp_[asettt]*al[L][a][b][aset]*al[L][c][d][astt]/(2.*L+1.);
						// bias[L][a][b][c][d][asette]=temp_[asette]*al[L][a][b][aset]*al[L][c][d][aste]/(2.*L+1.);
						// bias[L][a][b][c][d][asetet]=temp_[asetet]*al[L][a][b][aset]*al[L][c][d][aset]/(2.*L+1.);
						// bias[L][a][b][c][d][asetee]=temp_[asetee]*al[L][a][b][aset]*al[L][c][d][asee]/(2.*L+1.);
						// bias[L][a][b][c][d][aseett]=temp_[aseett]*al[L][a][b][asee]*al[L][c][d][astt]/(2.*L+1.);
						// bias[L][a][b][c][d][aseete]=temp_[aseete]*al[L][a][b][asee]*al[L][c][d][aste]/(2.*L+1.);
						// bias[L][a][b][c][d][aseeet]=temp_[aseeet]*al[L][a][b][asee]*al[L][c][d][aset]/(2.*L+1.);
						// bias[L][a][b][c][d][aseeee]=temp_[aseeee]*al[L][a][b][asee]*al[L][c][d][asee]/(2.*L+1.);
						// bias[L][a][b][c][d][astbtb]=temp_[astbtb]*al[L][a][b][astb]*al[L][c][d][astb]/(2.*L+1.);
						// bias[L][a][b][c][d][astbbt]=temp_[astbbt]*al[L][a][b][astb]*al[L][c][d][asbt]/(2.*L+1.);
						// bias[L][a][b][c][d][astbeb]=temp_[astbeb]*al[L][a][b][astb]*al[L][c][d][aseb]/(2.*L+1.);
						// bias[L][a][b][c][d][astbbe]=temp_[astbbe]*al[L][a][b][astb]*al[L][c][d][asbe]/(2.*L+1.);
						// bias[L][a][b][c][d][asbttb]=temp_[asbttb]*al[L][a][b][asbt]*al[L][c][d][astb]/(2.*L+1.);
						// bias[L][a][b][c][d][asbtbt]=temp_[asbtbt]*al[L][a][b][asbt]*al[L][c][d][asbt]/(2.*L+1.);
						// bias[L][a][b][c][d][asbteb]=temp_[asbteb]*al[L][a][b][asbt]*al[L][c][d][aseb]/(2.*L+1.);
						// bias[L][a][b][c][d][asbtbe]=temp_[asbtbe]*al[L][a][b][asbt]*al[L][c][d][asbe]/(2.*L+1.);
						// bias[L][a][b][c][d][asebtb]=temp_[asebtb]*al[L][a][b][aseb]*al[L][c][d][astb]/(2.*L+1.);
						// bias[L][a][b][c][d][asebbt]=temp_[asebbt]*al[L][a][b][aseb]*al[L][c][d][asbt]/(2.*L+1.);
						// bias[L][a][b][c][d][asebeb]=temp_[asebeb]*al[L][a][b][aseb]*al[L][c][d][aseb]/(2.*L+1.);
						// bias[L][a][b][c][d][asebbe]=temp_[asebbe]*al[L][a][b][aseb]*al[L][c][d][asbe]/(2.*L+1.);
						// bias[L][a][b][c][d][asbetb]=temp_[asbetb]*al[L][a][b][asbe]*al[L][c][d][astb]/(2.*L+1.);
						// bias[L][a][b][c][d][asbebt]=temp_[asbebt]*al[L][a][b][asbe]*al[L][c][d][asbt]/(2.*L+1.);
						// bias[L][a][b][c][d][asbeeb]=temp_[asbeeb]*al[L][a][b][asbe]*al[L][c][d][aseb]/(2.*L+1.);
						// bias[L][a][b][c][d][asbebe]=temp_[asbebe]*al[L][a][b][asbe]*al[L][c][d][asbe]/(2.*L+1.);
						bias[L][a][b][c][d][tttt]=temp_[tttt]*al[L][a][b][tt]*al[L][c][d][tt]/(2.*L+1.);
						bias[L][a][b][c][d][ttte]=temp_[ttte]*al[L][a][b][tt]*al[L][c][d][te]/(2.*L+1.);
						bias[L][a][b][c][d][ttee]=temp_[ttee]*al[L][a][b][tt]*al[L][c][d][ee]/(2.*L+1.);
						bias[L][a][b][c][d][tete]=temp_[tete]*al[L][a][b][te]*al[L][c][d][te]/(2.*L+1.);
						bias[L][a][b][c][d][teee]=temp_[teee]*al[L][a][b][te]*al[L][c][d][ee]/(2.*L+1.);
						bias[L][a][b][c][d][eeee]=temp_[eeee]*al[L][a][b][ee]*al[L][c][d][ee]/(2.*L+1.);
						bias[L][a][b][c][d][tbtb]=temp_[tbtb]*al[L][a][b][tb]*al[L][c][d][tb]/(2.*L+1.);
						bias[L][a][b][c][d][tbeb]=temp_[tbeb]*al[L][a][b][tb]*al[L][c][d][eb]/(2.*L+1.);
						bias[L][a][b][c][d][ebeb]=temp_[ebeb]*al[L][a][b][eb]*al[L][c][d][eb]/(2.*L+1.);
						bias[L][a][b][c][d][bbbb]=temp_[bbbb]*al[L][a][b][bb]*al[L][c][d][bb]/(2.*L+1.);

						PyErr_CheckSignals();
					}
				}
			}
		}
		std::cout << L << std::endl;
		PyErr_CheckSignals();
	}
	
	return bias;
}

void lensCls(PowSpec& llcl, PowSpec& ulcl, std::vector<double> &clDD) {
	int lmax_ul=ulcl.Lmax();
	int lmax_DD=clDD.size()-1;
	int lmax_ll=llcl.Lmax();
		
	std::vector< std::vector<double> > F0(lmax_DD+1, std::vector<double>(lmax_ul+1,0.));
	std::vector< std::vector<double> > F2(lmax_DD+1, std::vector<double>(lmax_ul+1,0.));
	
	for (size_t l1=0;l1<lmax_ll+1;l1++) {
		if (l1>=2) {
			compF(F0, l1, 0, lmax_DD+1, lmax_ul+1);
			compF(F2, l1, 2, lmax_DD+1, lmax_ul+1);
					
			double TTout=0.;
			double TEout=0.;
			double TBout=0.;
			double EEout=0.;
			double EBout=0.;
			double BBout=0.;
			#pragma omp parallel for reduction(+:TTout,TEout,TBout,EEout,EBout,BBout)
			for (size_t L=2;L<lmax_DD+1;L++) {
				for (size_t l2=2;l2<lmax_ul+1;l2++) {
					TTout+=F0[L][l2]*F0[L][l2]*clDD[L]*ulcl.tt(l2);
					TEout+=F0[L][l2]*F2[L][l2]*clDD[L]*ulcl.tg(l2);
					TBout+=F0[L][l2]*F2[L][l2]*clDD[L]*ulcl.tc(l2);
					if ((l1+L+l2)%2!=0) {
						EEout+=F2[L][l2]*F2[L][l2]*clDD[L]*ulcl.cc(l2);
						EBout+=-F2[L][l2]*F2[L][l2]*clDD[L]*ulcl.gc(l2);
						BBout+=F2[L][l2]*F2[L][l2]*clDD[L]*ulcl.gg(l2);
					}
					else{
						EEout+=F2[L][l2]*F2[L][l2]*clDD[L]*ulcl.gg(l2);
						EBout+=F2[L][l2]*F2[L][l2]*clDD[L]*ulcl.gc(l2);
						BBout+=F2[L][l2]*F2[L][l2]*clDD[L]*ulcl.cc(l2);
					}
				}
			}
		
			llcl.tt(l1)=TTout*1./(2.*l1+1.);
			llcl.tg(l1)=TEout*1./(2.*l1+1.);
			llcl.tc(l1)=TBout*1./(2.*l1+1.);
			llcl.gg(l1)=EEout*1./(2.*l1+1.);
			llcl.gc(l1)=EBout*1./(2.*l1+1.);
			llcl.cc(l1)=BBout*1./(2.*l1+1.);
		}
						
		if(PyErr_CheckSignals() == -1) {
			throw invalid_argument( "Keyboard interrupt" );
		}
				
	}
}

void systCls(PowSpec& llcl, PowSpec& ulcl, std::vector<double> &clDD, int type) {
	int lmax_ul=ulcl.Lmax();
	int lmax_DD=clDD.size()-1;
	int lmax_ll=llcl.Lmax();
	int lmax = max(lmax_ul,lmax_DD);
		
	std::vector< std::vector<double> > F(lmax+1, std::vector<double>(lmax+1,0.));
	std::vector< std::vector<double> > Fa(lmax+1, std::vector<double>(lmax+1,0.));
	
	for (size_t l1=0;l1<lmax_ll+1;l1++) {
		if (l1>=2) {
			
			if      (type==0) compF_a(F, l1, lmax+1);
			else if (type==1) compF_a(F, l1, lmax+1);
			else if (type==2) compF_f(F, l1, lmax+1);
			else if (type==3) compF_f(F, l1, lmax+1);
			else if (type==4) compF_gamma(F, l1, lmax+1);
			else if (type==5) compF_gamma(F, l1, lmax+1);
			else if (type==6){compF_pa(Fa, l1, lmax+1);compF_pb(F, l1, lmax+1);}
			else if (type==7){compF_pa(Fa, l1, lmax+1);compF_pb(F, l1, lmax+1);}
			else if (type==8) compF_d(F, l1, lmax+1);
			else if (type==9) compF_d(F, l1, lmax+1);
			else if (type==10)compF_q(F, l1, lmax+1);
			
			double BBout=0.;
			#pragma omp parallel for reduction(+:BBout)
			for (size_t L=2;L<lmax_DD+1;L++) {
				for (size_t l2=2;l2<lmax_ul+1;l2++) {
					if ((l1+L+l2)%2!=0) {
						if      (type==0) BBout+=F[L][l2]*F[L][l2]*clDD[L]*ulcl.gg(l2);
						else if (type==1) BBout+=-2*F[L][l2]*F[L][l2]*clDD[L]*ulcl.cc(l2);
						else if (type==2) BBout+=-F[L][l2]*F[L][l2]*clDD[L]*ulcl.gg(l2);
						else if (type==3) BBout+=-F[L][l2]*F[L][l2]*clDD[L]*ulcl.cc(l2);
						else if (type==4) BBout+=0;
						else if (type==5) BBout+=F[L][l2]*F[L][l2]*clDD[L]*ulcl.tt(l2);
						else if (type==6) BBout+=(Fa[L][l2]-F[L][l2])*(Fa[L][l2]-F[L][l2])*clDD[L]*ulcl.gg(l2);
						else if (type==7) BBout+=(Fa[L][l2]+F[L][l2])*(Fa[L][l2]+F[L][l2])*clDD[L]*ulcl.cc(l2);
						else if (type==8) BBout+=F[L][l2]*F[L][l2]*clDD[L]*ulcl.tt(l2);
						else if (type==8) BBout+=F[L][l2]*F[L][l2]*clDD[L]*ulcl.tt(l2);
						else if (type==10)BBout+=0;
					}
					else{
						if      (type==0) BBout+=F[L][l2]*F[L][l2]*clDD[L]*ulcl.cc(l2);
						else if (type==1) BBout+=2*F[L][l2]*F[L][l2]*clDD[L]*ulcl.gg(l2);
						else if (type==2) BBout+=F[L][l2]*F[L][l2]*clDD[L]*ulcl.cc(l2);
						else if (type==3) BBout+=F[L][l2]*F[L][l2]*clDD[L]*ulcl.gg(l2);
						else if (type==4) BBout+=F[L][l2]*F[L][l2]*clDD[L]*ulcl.tt(l2);
						else if (type==5) BBout+=0;
						else if (type==6) BBout+=(Fa[L][l2]+F[L][l2])*(Fa[L][l2]+F[L][l2])*clDD[L]*ulcl.cc(l2);
						else if (type==7) BBout+=(Fa[L][l2]-F[L][l2])*(Fa[L][l2]-F[L][l2])*clDD[L]*ulcl.gg(l2);
						else if (type==8) BBout+=F[L][l2]*F[L][l2]*clDD[L]*ulcl.tt(l2);
					    else if (type==9) BBout+=F[L][l2]*F[L][l2]*clDD[L]*ulcl.tt(l2);
					    else if (type==10)BBout+=0;
					}
				}
			}
		
			llcl.tt(l1)=BBout*1./(2.*l1+1.);
		}
						
		if(PyErr_CheckSignals() == -1) {
			throw invalid_argument( "Keyboard interrupt" );
		}
				
	}
}