#include <datatypes.h>


enum spectra {tt, te, ee, tb, eb, bb};
enum crosses {tttt, ttte, ttee, tete, teee, eeee, tbtb, tbeb, ebeb};
enum bh_type {grad, mask, nois, NUM_BH_TYPE};
enum bh_type_cross {gradgrad, gradmask, gradnois, maskmask, masknois, noisnois, NUM_BH_TYPE_CROSS};

int string2esttype(std::string inpu) {
	int type;
	if (inpu=="TT") type=tt;
	else if (inpu=="TE") type=te;
	else if (inpu=="EE") type=ee;
	else if (inpu=="TB") type=tb;
	else if (inpu=="EB") type=eb;
	else if (inpu=="BB") type=bb;
	else {
		type=-1;
		throw invalid_argument( "Unknown estimator!" );
	}
	return type;
}

xcomplex< double > complex_i(0.0,1.0);

double sgn(int m) {
	if (m==0) return (1.);
	else if (m%2==0) return (1.);
	else return (-1.);
}




void lensTT(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, int nside, arr<double> &weight);
void lensTE(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, int nside, arr<double> &weight);
void lensEE(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, int nside, arr<double> &weight);
void lensTB(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, int nside, arr<double> &weight);
void lensEB(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, int nside, arr<double> &weight);

void rotTT(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, int nside, arr<double> &weight);
void rotTE(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, int nside, arr<double> &weight);
void rotEE(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, int nside, arr<double> &weight);
void rotTB(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, int nside, arr<double> &weight);
void rotEB(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, int nside, arr<double> &weight);

void ampTT(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, int nside, arr<double> &weight);
void ampTE(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, int nside, arr<double> &weight);
void ampEE(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, int nside, arr<double> &weight);
void ampTB(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, int nside, arr<double> &weight);
void ampEB(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, int nside, arr<double> &weight);


void est_grad(Alm< xcomplex< double > > &alm1, Alm< xcomplex< double > > &alm2, std::string stype, Alm< xcomplex< double > > &almG, PowSpec& wcl, PowSpec& dcl, int lmin, int lminCMB1, int lminCMB2,  int lmaxCMB1, int lmaxCMB2, int nside);
void est_amp(Alm< xcomplex< double > > &alm1, Alm< xcomplex< double > > &alm2, std::string stype, Alm< xcomplex< double > > &almG, PowSpec& wcl, PowSpec& dcl, int lmin, int lminCMB1, int lminCMB2,  int lmaxCMB1, int lmaxCMB2, int nside);

void makeA_syst(PowSpec& wcl, PowSpec& dcl, PowSpec& al, size_t lmin, size_t lmax, size_t lminCMB, size_t type);

void map2alm_spin_iter(sharp_cxxjob<double> &job, Healpix_Map<double> &mapQ, Healpix_Map<double> &mapU, Alm<xcomplex<double> > &almG, Alm<xcomplex<double> > &almC, int spin, int num_iter);
void alm2map_spin(sharp_cxxjob<double> &job,  Alm<xcomplex<double> > &almG, Alm<xcomplex<double> > &almC, Healpix_Map<double> &mapQ, Healpix_Map<double> &mapU, int spin);
void alm2map_(sharp_cxxjob<double> &job,  Alm<xcomplex<double> > &almG, Healpix_Map<double> &mapQ);