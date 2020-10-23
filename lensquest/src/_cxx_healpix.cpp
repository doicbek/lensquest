#include <omp.h>


void map2alm_spin_iter(sharp_cxxjob<double> &job, Healpix_Map<double> &mapQ, Healpix_Map<double> &mapU, Alm<xcomplex<double> > &almG, Alm<xcomplex<double> > &almC, int spin, int num_iter) {
	size_t lmax=almG.Lmax();
	job.set_triangular_alm_info (lmax, lmax);	
	
	job.map2alm_spin(&mapQ[0],&mapU[0],&almG(0,0),&almC(0,0),spin,false);
	for (int iter=1; iter<=num_iter; ++iter) {
		Healpix_Map<double> mapQ2(mapQ.Nside(),mapQ.Scheme(),SET_NSIDE),mapU2(mapU.Nside(),mapU.Scheme(),SET_NSIDE);
		job.alm2map_spin(&almG(0,0),&almC(0,0),&mapQ2[0],&mapU2[0],spin,false);
		#pragma omp parallel for
		for (int m=0; m<mapQ.Npix(); ++m) {
			mapQ2[m] = mapQ[m]-mapQ2[m];
			mapU2[m] = mapU[m]-mapU2[m];
		}
		job.map2alm_spin(&mapQ2[0],&mapU2[0],&almG(0,0),&almC(0,0),spin,true);
	}
}

void alm2map_spin(sharp_cxxjob<double> &job,  Alm<xcomplex<double> > &almG, Alm<xcomplex<double> > &almC, Healpix_Map<double> &mapQ, Healpix_Map<double> &mapU, int spin) {
	size_t lmax=almG.Lmax();
	job.set_triangular_alm_info (lmax, lmax);	
	job.alm2map_spin(&almG(0,0),&almC(0,0),&mapQ[0],&mapU[0],spin,false);
}

void alm2map_(sharp_cxxjob<double> &job,  Alm<xcomplex<double> > &almG, Healpix_Map<double> &mapQ) {
	size_t lmax=almG.Lmax();
	job.set_triangular_alm_info (lmax, lmax);	
	job.alm2map(&almG(0,0), &mapQ[0], false);
}