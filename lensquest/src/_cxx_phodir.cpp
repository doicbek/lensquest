void dirTE(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, int nside, arr<double> &weight) {
	lensTE(alm1, alm2, almG, almC, wcl, nside, weight);
	arr< double > lsqrt;
	size_t lmax_almG=almG.Lmax();
	lsqrt.alloc(lmax_almG+1);
	for (size_t l=2; l<=lmax_almG; l++) {
		lsqrt[l]=1./sqrt(l*(l+1.));
	}
	#pragma omp parallel for
	for (size_t m=0; m<=lmax_almG; ++m) {
		for (size_t l=m; l<=lmax_almG; ++l) {
			almG(l,m)*=-1.*complex_i*lsqrt[l];
			almC(l,m)*=-1.*complex_i*lsqrt[l];
		}
	}
}

void dirTB(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, int nside, arr<double> &weight) {
	lensTB(alm1, alm2, almG, almC, wcl, nside, weight);
	arr< double > lsqrt;
	size_t lmax_almG=almG.Lmax();
	lsqrt.alloc(lmax_almG+1);
	for (size_t l=2; l<=lmax_almG; l++) {
		lsqrt[l]=1./sqrt(l*(l+1.));
	}
	#pragma omp parallel for
	for (size_t m=0; m<=lmax_almG; ++m) {
		for (size_t l=m; l<=lmax_almG; ++l) {
			almG(l,m)*=-1.*complex_i*lsqrt[l];
			almC(l,m)*=-1.*complex_i*lsqrt[l];
		}
	}
}

void dirEE(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, int nside, arr<double> &weight) {
	lensEE(alm1, alm2, almG, almC, wcl, nside, weight);
	arr< double > lsqrt;
	size_t lmax_almG=almG.Lmax();
	lsqrt.alloc(lmax_almG+1);
	for (size_t l=2; l<=lmax_almG; l++) {
		lsqrt[l]=1./sqrt(l*(l+1.));
	}
	#pragma omp parallel for
	for (size_t m=0; m<=lmax_almG; ++m) {
		for (size_t l=m; l<=lmax_almG; ++l) {
			almG(l,m)*=-1.*complex_i*lsqrt[l];
			almC(l,m)*=-1.*complex_i*lsqrt[l];
		}
	}
}

void dirEB(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, int nside, arr<double> &weight) {
	lensEB(alm1, alm2, almG, almC, wcl, nside, weight);
	arr< double > lsqrt;
	size_t lmax_almG=almG.Lmax();
	lsqrt.alloc(lmax_almG+1);
	for (size_t l=2; l<=lmax_almG; l++) {
		lsqrt[l]=1./sqrt(l*(l+1.));
	}
	#pragma omp parallel for
	for (size_t m=0; m<=lmax_almG; ++m) {
		for (size_t l=m; l<=lmax_almG; ++l) {
			almG(l,m)*=-1.*complex_i*lsqrt[l];
			almC(l,m)*=-1.*complex_i*lsqrt[l];
		}
	}
}