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
		case te: ampTE(alm1, alm2, almG, almC, wcl, nside, weight); break;
		case tb: ampTB(alm1, alm2, almG, almC, wcl, nside, weight); break;
		case ee: ampEE(alm1, alm2, almG, almC, wcl, nside, weight); break;
		case eb: ampEB(alm1, alm2, almG, almC, wcl, nside, weight); break;
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

void computef_systfb(std::vector< std::vector< std::vector<xcomplex< double >> > >& f, size_t L, PowSpec& wcl, size_t lminCMB, size_t lmaxCMB){
	
	std::vector< std::vector<double> > F;
	
	F=std::vector< std::vector<double> >(lmaxCMB+1, std::vector<double>(lmaxCMB+1,0.0));
	
	compF_f(F, L, lmaxCMB+1);
	
	#pragma omp parallel for
	for (size_t l1=lminCMB;l1<lmaxCMB+1;l1++) {
		for (size_t l3=lminCMB;l3<lmaxCMB+1;l3++) {
			if ((l1+L+l3)%2!=0) {
				f[0][l1][l3]=-wcl.tg(l1)*F[l3][l1];
				f[1][l1][l3]=-wcl.gg(l1)*F[l3][l1]+wcl.gg(l3)*F[l1][l3];
				f[2][l1][l3]=0.0; 
				f[3][l1][l3]=0.0;
			}
			else {
				f[0][l1][l3]=0.0; 
				f[1][l1][l3]=0.0; 
				f[2][l1][l3]=wcl.tg(l1)*F[l3][l1];
				f[3][l1][l3]=wcl.gg(l1)*F[l3][l1]-wcl.cc(l3)*F[l1][l3];
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
				f[0][l1][l3]=wcl.tt(l1)*F[l3][l1];
				f[1][l1][l3]=wcl.tg(l1)*F[l3][l1]-wcl.tg(l3)*F[l1][l3];
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

void computef_systdb(std::vector< std::vector< std::vector<xcomplex< double >> > >& f, size_t L, PowSpec& wcl, size_t lminCMB, size_t lmaxCMB){
	
	std::vector< std::vector<double> > F;
	
	F=std::vector< std::vector<double> >(lmaxCMB+1, std::vector<double>(lmaxCMB+1,0.0));
	
	compF_d(F, L, lmaxCMB+1);
	
	#pragma omp parallel for
	for (size_t l1=lminCMB;l1<lmaxCMB+1;l1++) {
		for (size_t l3=lminCMB;l3<lmaxCMB+1;l3++) {
			if ((l1+L+l3)%2==0) {
				f[0][l1][l3]=wcl.tt(l1)*F[l3][l1];
				f[1][l1][l3]=wcl.tg(l1)*F[l3][l1]-wcl.tg(l3)*F[l1][l3];
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


void makeA_syst(PowSpec& wcl, PowSpec& dcl, PowSpec& al, size_t lmin, size_t lmax, size_t lminCMB, int type) {
	size_t lmaxCMB=dcl.Lmax();
			
	std::vector< std::vector< std::vector<xcomplex< double >> > > f(4, std::vector< std::vector<xcomplex< double >> >(lmaxCMB+1, std::vector<xcomplex< double >>(lmaxCMB+1,0.0)));

	std::vector<double> invlcltt(lmaxCMB+1,0.0), invlclee(lmaxCMB+1,0.0), invlclbb(lmaxCMB+1,0.0);
	
	#pragma omp parallel for
	for (size_t l1=lminCMB;l1<lmaxCMB+1;l1++) {
		invlcltt[l1]=1./dcl.tt(l1);
		invlclee[l1]=1./dcl.gg(l1);
		invlclbb[l1]=1./dcl.cc(l1);
	}
	
	double ate, aee, atb, aeb;
	
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
		
		if (L>=lmin) {
			#pragma omp parallel for reduction(+:ate, aee, atb, aeb) schedule(dynamic, 25)
			for (size_t l1=lminCMB;l1<lmaxCMB+1;l1++) {
				for (size_t l3=lminCMB;l3<lmaxCMB+1;l3++) {
					ate+=real(f[0][l1][l3]*conj(f[0][l1][l3]))*invlcltt[l1]*invlclee[l3];
					aee+=real(f[1][l1][l3]*conj(f[1][l1][l3]))*invlclee[l1]*invlclee[l3]*.5;
					atb+=real(f[2][l1][l3]*conj(f[2][l1][l3]))*invlcltt[l1]*invlclbb[l3]; 
					aeb+=real(f[3][l1][l3]*conj(f[3][l1][l3]))*invlclee[l1]*invlclbb[l3];
				}
			}
		}
		
		al.tg(L) = (ate!=0.) ? (2.0*L+1.0)/ate : 0.0;
		al.gg(L) = (aee!=0.) ? (2.0*L+1.0)/aee : 0.0;
		al.tc(L) = (atb!=0.) ? (2.0*L+1.0)/atb : 0.0;
		al.gc(L) = (aeb!=0.) ? (2.0*L+1.0)/aeb : 0.0;
		
		if(PyErr_CheckSignals() == -1) {
			throw invalid_argument( "Keyboard interrupt" );
		}
	}
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