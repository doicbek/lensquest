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


void computef_systa(std::vector< std::vector< std::vector<double> > >& f, size_t L, PowSpec& wcl, size_t lminCMB, size_t lmaxCMB){
	
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

void computef_systomega(std::vector< std::vector< std::vector<double> > >& f, size_t L, PowSpec& wcl, size_t lminCMB, size_t lmaxCMB){
	
	std::vector< std::vector<double> > F;
	
	F=std::vector< std::vector<double> >(lmaxCMB+1, std::vector<double>(lmaxCMB+1,0.0));
	
	compF_a(F, L, lmaxCMB+1);
	
	#pragma omp parallel for
	for (size_t l1=lminCMB;l1<lmaxCMB+1;l1++) {
		for (size_t l3=lminCMB;l3<lmaxCMB+1;l3++) {
			if ((l1+L+l3)%2!=0) {
				f[0][l1][l3]=-2*wcl.tg(l1)*F[l3][l1];
				f[1][l1][l3]=-2*wcl.gg(l1)*F[l3][l1]-2*wcl.gg(l3)*F[l1][l3];
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

void computef_systfa(std::vector< std::vector< std::vector<double> > >& f, size_t L, PowSpec& wcl, size_t lminCMB, size_t lmaxCMB){
	
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

void computef_systfb(std::vector< std::vector< std::vector<double> > >& f, size_t L, PowSpec& wcl, size_t lminCMB, size_t lmaxCMB){
	
	std::vector< std::vector<double> > F;
	
	F=std::vector< std::vector<double> >(lmaxCMB+1, std::vector<double>(lmaxCMB+1,0.0));
	
	compF_f(F, L, lmaxCMB+1);
	
	#pragma omp parallel for
	for (size_t l1=lminCMB;l1<lmaxCMB+1;l1++) {
		for (size_t l3=lminCMB;l3<lmaxCMB+1;l3++) {
			if ((l1+L+l3)%2!=0) {
				f[0][l1][l3]=-wcl.tg(l1)*F[l3][l1];
				f[1][l1][l3]=-wcl.gg(l1)*F[l3][l1]-wcl.gg(l3)*F[l1][l3];
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

void computef_systgammaa(std::vector< std::vector< std::vector<double> > >& f, size_t L, PowSpec& wcl, size_t lminCMB, size_t lmaxCMB){
	
	std::vector< std::vector<double> > F;
	
	F=std::vector< std::vector<double> >(lmaxCMB+1, std::vector<double>(lmaxCMB+1,0.0));
	
	compF_gamma(F, L, lmaxCMB+1);
	
	#pragma omp parallel for
	for (size_t l1=lminCMB;l1<lmaxCMB+1;l1++) {
		for (size_t l3=lminCMB;l3<lmaxCMB+1;l3++) {
			if ((l1+L+l3)%2==0) {
				f[0][l1][l3]=0;
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

void computef_systgammab(std::vector< std::vector< std::vector<double> > >& f, size_t L, PowSpec& wcl, size_t lminCMB, size_t lmaxCMB){
	
	std::vector< std::vector<double> > F;
	
	F=std::vector< std::vector<double> >(lmaxCMB+1, std::vector<double>(lmaxCMB+1,0.0));
	
	compF_gamma(F, L, lmaxCMB+1);
	
	#pragma omp parallel for
	for (size_t l1=lminCMB;l1<lmaxCMB+1;l1++) {
		for (size_t l3=lminCMB;l3<lmaxCMB+1;l3++) {
			if ((l1+L+l3)%2!=0) {
				f[0][l1][l3]=0;
				f[1][l1][l3]=-wcl.tg(l1)*F[l3][l1]-wcl.tg(l3)*F[l1][l3];
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

void computef_systpa(std::vector< std::vector< std::vector<double> > >& f, size_t L, PowSpec& wcl, size_t lminCMB, size_t lmaxCMB){
	
	std::vector< std::vector<double> > Fa, Fb;
	
	Fa=std::vector< std::vector<double> >(lmaxCMB+1, std::vector<double>(lmaxCMB+1,0.0));
	Fb=std::vector< std::vector<double> >(lmaxCMB+1, std::vector<double>(lmaxCMB+1,0.0));
	
	compF_pa(Fa, L, lmaxCMB+1);
	compF_pb(Fb, L, lmaxCMB+1);
	
	#pragma omp parallel for
	for (size_t l1=lminCMB;l1<lmaxCMB+1;l1++) {
		for (size_t l3=lminCMB;l3<lmaxCMB+1;l3++) {
			f[0][l1][l3]=wcl.tg(l1)*(Fa[l3][l1]+Fb[l3][l1]); 
			f[1][l1][l3]=wcl.gg(l3)*(Fa[l1][l3]+Fb[l1][l3])+wcl.gg(l1)*(Fa[l3][l1]+Fb[l3][l1]); 
			f[2][l1][l3]=-wcl.tg(l1)*(Fa[l3][l1]-Fb[l3][l1]);
			f[3][l1][l3]=-wcl.gg(l1)*(Fa[l3][l1]-Fb[l3][l1])+wcl.cc(l3)*(Fa[l1][l3]-Fb[l1][l3]);
		}
	}
}

void computef_systpb(std::vector< std::vector< std::vector<double> > >& f, size_t L, PowSpec& wcl, size_t lminCMB, size_t lmaxCMB){
	
	std::vector< std::vector<double> > Fa, Fb;
	
	Fa=std::vector< std::vector<double> >(lmaxCMB+1, std::vector<double>(lmaxCMB+1,0.0));
	Fb=std::vector< std::vector<double> >(lmaxCMB+1, std::vector<double>(lmaxCMB+1,0.0));
	
	compF_pa(Fa, L, lmaxCMB+1);
	compF_pb(Fb, L, lmaxCMB+1);
	
	#pragma omp parallel for
	for (size_t l1=lminCMB;l1<lmaxCMB+1;l1++) {
		for (size_t l3=lminCMB;l3<lmaxCMB+1;l3++) {
			f[0][l1][l3]=wcl.tg(l1)*(Fa[l3][l1]-Fb[l3][l1]); 
			f[1][l1][l3]=wcl.gg(l3)*(Fa[l1][l3]-Fb[l1][l3])+wcl.gg(l1)*(Fa[l3][l1]-Fb[l3][l1]); 
			f[2][l1][l3]=-wcl.tg(l1)*(Fa[l3][l1]+Fb[l3][l1]);
			f[3][l1][l3]=-wcl.gg(l1)*(Fa[l3][l1]+Fb[l3][l1])+wcl.cc(l3)*(Fa[l1][l3]+Fb[l1][l3]);
		}
	}
}

void computef_systda(std::vector< std::vector< std::vector<double> > >& f, size_t L, PowSpec& wcl, size_t lminCMB, size_t lmaxCMB){
	
	std::vector< std::vector<double> > F;
	
	F=std::vector< std::vector<double> >(lmaxCMB+1, std::vector<double>(lmaxCMB+1,0.0));
	
	compF_d(F, L, lmaxCMB+1);
	
	#pragma omp parallel for
	for (size_t l1=lminCMB;l1<lmaxCMB+1;l1++) {
		for (size_t l3=lminCMB;l3<lmaxCMB+1;l3++) {
			f[0][l1][l3]=0;
			f[1][l1][l3]=sgn(l1+L+l3)*wcl.tg(l1)*F[l3][l1]+wcl.tg(l3)*F[l1][l3];
			f[2][l1][l3]=wcl.tt(l1)*F[l3][l1];
			f[3][l1][l3]=wcl.tg(l1)*F[l3][l1];
		}
	}
}

void computef_systdb(std::vector< std::vector< std::vector<double> > >& f, size_t L, PowSpec& wcl, size_t lminCMB, size_t lmaxCMB){
	
	std::vector< std::vector<double> > F;
	
	F=std::vector< std::vector<double> >(lmaxCMB+1, std::vector<double>(lmaxCMB+1,0.0));
	
	compF_d(F, L, lmaxCMB+1);
	
	#pragma omp parallel for
	for (size_t l1=lminCMB;l1<lmaxCMB+1;l1++) {
		for (size_t l3=lminCMB;l3<lmaxCMB+1;l3++) {
			f[0][l1][l3]=0;
			f[1][l1][l3]=wcl.tg(l1)*F[l3][l1]+sgn(l1+L+l3)*wcl.tg(l3)*F[l1][l3];
			f[2][l1][l3]=sgn(l1+L+l3)*wcl.tt(l1)*F[l3][l1];
			f[3][l1][l3]=sgn(l1+L+l3)*wcl.tg(l1)*F[l3][l1];
		}
	}
}

void computef_systq(std::vector< std::vector< std::vector<double> > >& f, size_t L, PowSpec& wcl, size_t lminCMB, size_t lmaxCMB){
	
	std::vector< std::vector<double> > F;
	
	F=std::vector< std::vector<double> >(lmaxCMB+1, std::vector<double>(lmaxCMB+1,0.0));
	
	compF_q(F, L, lmaxCMB+1);
	
	#pragma omp parallel for
	for (size_t l1=lminCMB;l1<lmaxCMB+1;l1++) {
		for (size_t l3=lminCMB;l3<lmaxCMB+1;l3++) {
			f[0][l1][l3]=0;
			f[1][l1][l3]=wcl.tg(l1)*F[l3][l1]+sgn(l1+L+l3)*wcl.tg(l3)*F[l1][l3];
			f[2][l1][l3]=sgn(l1+L+l3)*wcl.tt(l1)*F[l3][l1];
			f[3][l1][l3]=sgn(l1+L+l3)*wcl.tg(l1)*F[l3][l1];
		}
	}
}

void makeA_syst(PowSpec& wcl, PowSpec& dcl, PowSpec& al, size_t lmin, size_t lmax, size_t lminCMB, int type) {
	size_t lmaxCMB=dcl.Lmax();
			
	std::vector< std::vector< std::vector<double> > > f(4, std::vector< std::vector<double> >(lmaxCMB+1, std::vector<double>(lmaxCMB+1,0.0)));

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
					ate+=f[0][l1][l3]*f[0][l1][l3]*invlcltt[l1]*invlclee[l3];
					aee+=f[1][l1][l3]*f[1][l1][l3]*invlclee[l1]*invlclee[l3]*.5;
					atb+=f[2][l1][l3]*f[2][l1][l3]*invlcltt[l1]*invlclbb[l3]; 
					aeb+=f[3][l1][l3]*f[3][l1][l3]*invlclee[l1]*invlclbb[l3];
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