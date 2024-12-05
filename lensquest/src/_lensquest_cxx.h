#include <datatypes.h>


enum spectra {tt, te, ee, tb, eb, bb};
enum crosses {tttt, ttte, ttee, tete, teee, eeee, tbtb, tbeb, ebeb, bbbb};
enum bh_type {grad, mask, nois, NUM_BH_TYPE};
enum bh_type_cross {gradgrad, gradmask, gradnois, maskmask, masknois, noisnois, NUM_BH_TYPE_CROSS};
enum syst_type {syst_a, syst_o, syst_g1, syst_g2, syst_f1, syst_f2, syst_p1, syst_p2, syst_d1, syst_d2, syst_q};

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
void lensBB(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, int nside, arr<double> &weight);

void rotTE(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, int nside, arr<double> &weight);
void rotEE(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, int nside, arr<double> &weight);
void rotTB(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, int nside, arr<double> &weight);
void rotEB(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, int nside, arr<double> &weight);

void ampTT(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, int nside, arr<double> &weight);
void ampTE(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, int nside, arr<double> &weight);
void ampEE(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, int nside, arr<double> &weight);
void ampTB(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, int nside, arr<double> &weight);
void ampEB(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, int nside, arr<double> &weight);
void ampBB(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, int nside, arr<double> &weight);

void dbetaTT(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, std::vector<double>& rl1, std::vector<double>& rl2, int nside, arr<double> &weight);
void dbetaTE(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, std::vector<double>& rl1, std::vector<double>& rl2, int nside, arr<double> &weight);
void dbetaEE(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, std::vector<double>& rl1, std::vector<double>& rl2, int nside, arr<double> &weight);
void dbetaTB(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, std::vector<double>& rl1, std::vector<double>& rl2, int nside, arr<double> &weight);
void dbetaEB(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, std::vector<double>& rl1, std::vector<double>& rl2, int nside, arr<double> &weight);
void dbetaBB(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, std::vector<double>& rl1, std::vector<double>& rl2, int nside, arr<double> &weight);

void spiTE(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, int nside, arr<double> &weight);
void spiEE(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, int nside, arr<double> &weight);
void spiTB(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, int nside, arr<double> &weight);
void spiEB(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, int nside, arr<double> &weight);

void lmoTE(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, int nside, arr<double> &weight);
void lmoEE(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, int nside, arr<double> &weight);
void lmoTB(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, int nside, arr<double> &weight);
void lmoEB(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, int nside, arr<double> &weight);

void dirTE(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, int nside, arr<double> &weight);
void dirEE(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, int nside, arr<double> &weight);
void dirTB(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, int nside, arr<double> &weight);
void dirEB(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, int nside, arr<double> &weight);

void ldiTE(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, int nside, arr<double> &weight);
void ldiEE(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, int nside, arr<double> &weight);
void ldiTB(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, int nside, arr<double> &weight);
void ldiEB(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, int nside, arr<double> &weight);

void lquTE(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, int nside, arr<double> &weight);
void lquEE(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, int nside, arr<double> &weight);
void lquTB(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, int nside, arr<double> &weight);
void lquEB(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, int nside, arr<double> &weight);

void est_grad(Alm< xcomplex< double > > &alm1, Alm< xcomplex< double > > &alm2, std::string stype, Alm< xcomplex< double > > &almG, PowSpec& wcl, PowSpec& dcl, int lmin, int lminCMB1, int lminCMB2,  int lmaxCMB1, int lmaxCMB2, int nside);
void est_amp(Alm< xcomplex< double > > &alm1, Alm< xcomplex< double > > &alm2, std::string stype, Alm< xcomplex< double > > &almG, PowSpec& wcl, PowSpec& dcl, int lmin, int lminCMB1, int lminCMB2,  int lmaxCMB1, int lmaxCMB2, int nside);
void est_dbeta(Alm< xcomplex< double > > &alm1, Alm< xcomplex< double > > &alm2, std::string stype, Alm< xcomplex< double > > &almG, PowSpec& wcl, PowSpec& dcl, std::vector<double>& R1, std::vector<double>& R2, int lmin, int lminCMB1, int lminCMB2,  int lmaxCMB1, int lmaxCMB2, int nside);
void est_rot(Alm< xcomplex< double > > &alm1, Alm< xcomplex< double > > &alm2, std::string stype, Alm< xcomplex< double > > &almG, PowSpec& wcl, PowSpec& dcl, int lmin, int lminCMB1, int lminCMB2,  int lmaxCMB1, int lmaxCMB2, int nside);
void est_spinflip(Alm< xcomplex< double > > &alm1, Alm< xcomplex< double > > &alm2, std::string stype, Alm< xcomplex< double > > &almG, Alm< xcomplex< double > > &almC, PowSpec& wcl, PowSpec& dcl, int lmin, int lminCMB1, int lminCMB2,  int lmaxCMB1, int lmaxCMB2, int nside);
void est_monoleak(Alm< xcomplex< double > > &alm1, Alm< xcomplex< double > > &alm2, std::string stype, Alm< xcomplex< double > > &almG, Alm< xcomplex< double > > &almC, PowSpec& wcl, PowSpec& dcl, int lmin, int lminCMB1, int lminCMB2,  int lmaxCMB1, int lmaxCMB2, int nside);
void est_phodir(Alm< xcomplex< double > > &alm1, Alm< xcomplex< double > > &alm2, std::string stype, Alm< xcomplex< double > > &almG, Alm< xcomplex< double > > &almC, PowSpec& wcl, PowSpec& dcl, int lmin, int lminCMB1, int lminCMB2,  int lmaxCMB1, int lmaxCMB2, int nside);
void est_dipleak(Alm< xcomplex< double > > &alm1, Alm< xcomplex< double > > &alm2, std::string stype, Alm< xcomplex< double > > &almG, Alm< xcomplex< double > > &almC, PowSpec& wcl, PowSpec& dcl, int lmin, int lminCMB1, int lminCMB2,  int lmaxCMB1, int lmaxCMB2, int nside);
void est_quadleak(Alm< xcomplex< double > > &alm1, Alm< xcomplex< double > > &alm2, std::string stype, Alm< xcomplex< double > > &almG, Alm< xcomplex< double > > &almC, PowSpec& wcl, PowSpec& dcl, int lmin, int lminCMB1, int lminCMB2,  int lmaxCMB1, int lmaxCMB2, int nside);

std::vector< std::vector<double> > makeAN(PowSpec& wcl, PowSpec& dcl, PowSpec& rdcls, PowSpec& al, size_t lmin, size_t lmax, size_t lminCMB1, size_t lminCMB2, size_t lmaxCMB1, size_t lmaxCMB2);
std::vector< std::vector<double> > makeA_syst(PowSpec& wcl, PowSpec& dcl, PowSpec& rdcls, PowSpec& al, size_t lmin, size_t lmax, size_t lminCMB, size_t type);
std::vector< std::vector< std::vector< std::vector<double> > > > makeAN_syst(PowSpec& wcl, PowSpec& dcl, size_t lmin, size_t lmax, size_t lminCMB1, size_t lminCMB2, size_t lmaxCMB1, size_t lmaxCMB2);
std::vector< std::vector<double> > makeA_dust(PowSpec& wcl, PowSpec& dcl1, PowSpec& dcl2, PowSpec& R1, PowSpec& R2, PowSpec& rdcls, PowSpec& al, size_t lmin, size_t lmax, size_t lminCMB);
std::vector< std::vector< std::vector< std::vector< std::vector<std::vector< double > > > > > > makeX_dust(PowSpec &wcl, std::vector< std::vector< std::vector<double> > > & dcl, std::vector< std::vector< std::vector< std::vector<double> > > > & rd, std::vector<std::vector< std::vector< std::vector<double> > > >& al, std::vector<std::vector<std::vector<std::vector<double>>>>& rlnu, size_t lmin, size_t lmax, size_t lminCMB1, size_t lminCMB2, size_t lmaxCMB1, size_t lmaxCMB2);

void map2alm_spin_iter(sharp_cxxjob<double> &job, Healpix_Map<double> &mapQ, Healpix_Map<double> &mapU, Alm<xcomplex<double> > &almG, Alm<xcomplex<double> > &almC, int spin, int num_iter);
void map2alm_spin_iter(Healpix_Map<double> &mapQ, Healpix_Map<double> &mapU, Alm<xcomplex<double> > &almG, Alm<xcomplex<double> > &almC, int spin, int num_iter);
void alm2map_spin(sharp_cxxjob<double> &job,  Alm<xcomplex<double> > &almG, Alm<xcomplex<double> > &almC, Healpix_Map<double> &mapQ, Healpix_Map<double> &mapU, int spin);
void alm2map_spin(Alm<xcomplex<double> > &almG, Alm<xcomplex<double> > &almC, Healpix_Map<double> &mapQ, Healpix_Map<double> &mapU, int spin);
void alm2map_(sharp_cxxjob<double> &job,  Alm<xcomplex<double> > &almG, Healpix_Map<double> &mapQ);

void lensCls(PowSpec& llcl, PowSpec& ulcl, std::vector<double> &clDD);
void systCls(PowSpec& llcl, PowSpec& ulcl, std::vector<double> &clDD, int type);

std::vector< std::vector<double> > computeKernel(std::string stype, PowSpec& wcl, PowSpec& dcl, size_t lminCMB, size_t L);