void lensTT(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, int nside, arr<double> &weight) {
	size_t lmax_alm1=alm1.Lmax();
	size_t lmax_alm2=alm2.Lmax();
	size_t lmax_almG=almG.Lmax();
	
	arr<double> resp, lsqr;
	double tempdouble;

	almG.SetToZero();
	almC.SetToZero();
	
	sharp_cxxjob<double> job;
	job.set_weighted_Healpix_geometry (nside, &weight[0]);
	
	Healpix_Map<double> map1Q,map1U,map2Q,map2U;
	map1Q.SetNside(nside,RING);
	map1U.SetNside(nside,RING);
	map2Q.SetNside(nside,RING);
	map2U.SetNside(nside,RING);
	
	alm2map_(job,alm1,map1Q);

	resp.alloc(lmax_alm2+1);

	#pragma omp parallel for
	for (size_t l=2; l<=lmax_alm2; l++) {
		resp[l]=wcl.tt(l)*sqrt(l*(l+1.));
	}
	
	almG.Set(lmax_alm2, lmax_alm2);
	#pragma omp parallel for
	for (size_t m=0; m<=lmax_alm2; ++m) {
		for (size_t l=m; l<=lmax_alm2; ++l) {
			almG(l,m)=resp[l]*alm2(l,m);
		}
	} 

	job.set_triangular_alm_info (lmax_alm2, lmax_alm2);	
	almG.Set(lmax_alm2, lmax_alm2);
	almC.Set(lmax_alm2, lmax_alm2);
	alm2map_spin(job,almG,almC,map2Q,map2U,1);

	#pragma omp parallel for
	for (int i=0; i< map1Q.Npix(); i++) {
		tempdouble=map1Q[i]*1.;
		map1Q[i]=map1Q[i]*map2Q[i];
		map1U[i]=tempdouble*map2U[i];
	}

	almG.Set(lmax_almG, lmax_almG);
	almC.Set(lmax_almG, lmax_almG);
	map2alm_spin_iter(job,map1Q,map1U,almG,almC,1,3);
	
	lsqr.alloc(lmax_almG+1);
	#pragma omp parallel for
	for (size_t l=2; l<=lmax_almG; l++) {
		lsqr[l]=sqrt(l*(l+1.));
	}
	
	almG.ScaleL(lsqr);
	almC.ScaleL(lsqr);
}

void lensTE(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, int nside, arr<double> &weight) {
	size_t lmax_alm1=alm1.Lmax();
	size_t lmax_alm2=alm2.Lmax();
	size_t lmax_almG=almG.Lmax();
	
	arr<double> resp1, resp2, lsqr;
	double tempdouble;

	almG.SetToZero();
	almC.SetToZero();
	
	Alm< xcomplex< double > > almZ(lmax_almG,lmax_almG);
	almZ.SetToZero();
	
	sharp_cxxjob<double> job;
	job.set_weighted_Healpix_geometry (nside, &weight[0]);
	
	Healpix_Map<double> map1Q,map1U,map2Q,map2U,map3Q,map3U;
	map1Q.SetNside(nside,RING);
	map1U.SetNside(nside,RING);
	map2Q.SetNside(nside,RING);
	map2U.SetNside(nside,RING);
	map3Q.SetNside(nside,RING);
	map3U.SetNside(nside,RING);
	
	almZ.Set(lmax_alm2, lmax_alm2);
    alm2map_spin(job,alm2,almZ,map2Q,map2U,2);
	
	resp1.alloc(lmax_alm1+1);
	resp2.alloc(lmax_alm1+1);
	#pragma omp parallel for
	for (size_t l=2; l<=lmax_alm1; l++) {
		resp1[l]=wcl.tg(l)*sqrt((l+2.)*(l-1.));
		resp2[l]=wcl.tg(l)*sqrt((l-2.)*(l+3.));
	}
	
	almG.Set(lmax_alm1, lmax_alm1);
	almC.Set(lmax_alm1, lmax_alm1);
	#pragma omp parallel for
	for (size_t m=0; m<=lmax_alm1; ++m) {
		for (size_t l=m; l<=lmax_alm1; ++l) {
			almG(l,m)=resp1[l]*alm1(l,m);
			almC(l,m)=resp2[l]*alm1(l,m);
		}
	} 
	
	almZ.Set(lmax_alm1, lmax_alm1);
	alm2map_spin(job,almG,almZ,map1Q,map1U,1);
	alm2map_spin(job,almC,almZ,map3Q,map3U,3);
	
	#pragma omp parallel for
	for (int i=0; i< map1Q.Npix(); i++) {
		tempdouble=map1Q[i]*1.;
		map1Q[i]=map2Q[i]*(map1Q[i]-map3Q[i]) + map2U[i]*(map1U[i]-map3U[i]);
		map1U[i]=-map2Q[i]*(map1U[i]+map3U[i]) + map2U[i]*(tempdouble+map3Q[i]);
	}

	alm2map_(job,alm1,map2Q);

	resp1.alloc(lmax_alm2+1);
	#pragma omp parallel for
	for (size_t l=2; l<=lmax_alm2; l++) {
		resp1[l]=wcl.tg(l)*sqrt((l+1.)*l);
	}
	
	almG.Set(lmax_alm2, lmax_alm2);
	#pragma omp parallel for
	for (size_t m=0; m<=lmax_alm2; ++m) {
		for (size_t l=m; l<=lmax_alm2; ++l) {
			almG(l,m)=resp1[l]*alm2(l,m);
		}
	}
	
	almZ.Set(lmax_almG, lmax_almG);
	alm2map_spin(job,almG,almZ,map3Q,map3U,1);
	
	#pragma omp parallel for
	for (int i=0; i< map1Q.Npix(); i++) {
		map1Q[i]+=2*map2Q[i]*map3Q[i];
		map1U[i]+=2*map2Q[i]*map3U[i];
	}
	
	almG.Set(lmax_almG, lmax_almG);
	almC.Set(lmax_almG, lmax_almG);
	map2alm_spin_iter(job,map1Q,map1U,almG,almC,1,3);
	
	lsqr.alloc(lmax_almG+1);
	#pragma omp parallel for
	for (size_t l=2; l<=lmax_almG; l++) {
		lsqr[l]=sqrt(l*(l+1.))*.5;
	}
	
	almG.ScaleL(lsqr);
	almC.ScaleL(lsqr);
}

void lensTB(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, int nside, arr<double> &weight) {
	size_t lmax_alm1=alm1.Lmax();
	size_t lmax_alm2=alm2.Lmax();
	size_t lmax_almG=almG.Lmax();
	
	arr<double> resp1, resp2, lsqr;
	double tempdouble;

	almG.SetToZero();
	almC.SetToZero();
	
	Alm< xcomplex< double > > almZ(lmax_almG,lmax_almG);
	almZ.SetToZero();
	
	sharp_cxxjob<double> job;
	job.set_weighted_Healpix_geometry (nside, &weight[0]);
	
	Healpix_Map<double> map1Q,map1U,map2Q,map2U,map3Q,map3U;
	map1Q.SetNside(nside,RING);
	map1U.SetNside(nside,RING);
	map2Q.SetNside(nside,RING);
	map2U.SetNside(nside,RING);
	map3Q.SetNside(nside,RING);
	map3U.SetNside(nside,RING);
	
	almZ.Set(lmax_alm2, lmax_alm2);
    alm2map_spin(job,almZ,alm2,map2Q,map2U,2);
	
	resp1.alloc(lmax_alm1+1);
	resp2.alloc(lmax_alm1+1);
	#pragma omp parallel for
	for (size_t l=2; l<=lmax_alm1; l++) {
		resp1[l]=wcl.tg(l)*sqrt((l+2.)*(l-1.));
		resp2[l]=wcl.tg(l)*sqrt((l-2.)*(l+3.));
	}
	
	almG.Set(lmax_alm1, lmax_alm1);
	almC.Set(lmax_alm1, lmax_alm1);
	#pragma omp parallel for
	for (size_t m=0; m<=lmax_alm1; ++m) {
		for (size_t l=m; l<=lmax_alm1; ++l) {
			almG(l,m)=resp1[l]*alm1(l,m);
			almC(l,m)=resp2[l]*alm1(l,m);
		}
	} 
	
	almZ.Set(lmax_alm1, lmax_alm1);
	alm2map_spin(job,almG,almZ,map1Q,map1U,1);
	alm2map_spin(job,almC,almZ,map3Q,map3U,3);
	
	#pragma omp parallel for
	for (int i=0; i< map1Q.Npix(); i++) {
		tempdouble=map1Q[i]*1.;
		map1Q[i]=map2Q[i]*(map1Q[i]-map3Q[i]) + map2U[i]*(map1U[i]-map3U[i]);
		map1U[i]=-map2Q[i]*(map1U[i]+map3U[i]) + map2U[i]*(tempdouble+map3Q[i]);
	}
	
	almG.Set(lmax_almG, lmax_almG);
	almC.Set(lmax_almG, lmax_almG);
	map2alm_spin_iter(job,map1Q,map1U,almG,almC,1,3);
	
	lsqr.alloc(lmax_almG+1);
	#pragma omp parallel for
	for (size_t l=2; l<=lmax_almG; l++) {
		lsqr[l]=sqrt(l*(l+1.))*.5;
	}
	
	almG.ScaleL(lsqr);
	almC.ScaleL(lsqr);
}

void lensEE(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, int nside, arr<double> &weight) {
	size_t lmax_alm1=alm1.Lmax();
	size_t lmax_alm2=alm2.Lmax();
	size_t lmax_almG=almG.Lmax();
	
	arr<double> resp1, resp2, lsqr;
	double tempdouble;

	almG.SetToZero();
	almC.SetToZero();
	
	Alm< xcomplex< double > > almZ(lmax_almG,lmax_almG);
	almZ.SetToZero();
	
	sharp_cxxjob<double> job;
	job.set_weighted_Healpix_geometry (nside, &weight[0]);
	
	Healpix_Map<double> map1Q,map1U,map2Q,map2U,map3Q,map3U;
	map1Q.SetNside(nside,RING);
	map1U.SetNside(nside,RING);
	map2Q.SetNside(nside,RING);
	map2U.SetNside(nside,RING);
	map3Q.SetNside(nside,RING);
	map3U.SetNside(nside,RING);
	
	almZ.Set(lmax_alm2, lmax_alm2);
    alm2map_spin(job,alm2,almZ,map2Q,map2U,2);
	
	resp1.alloc(lmax_alm1+1);
	resp2.alloc(lmax_alm1+1);
	#pragma omp parallel for
	for (size_t l=2; l<=lmax_alm1; l++) {
		resp1[l]=wcl.gg(l)*sqrt((l+2.)*(l-1.));
		resp2[l]=wcl.gg(l)*sqrt((l-2.)*(l+3.));
	}
	
	almG.Set(lmax_alm1, lmax_alm1);
	almC.Set(lmax_alm1, lmax_alm1);
	#pragma omp parallel for
	for (size_t m=0; m<=lmax_alm1; ++m) {
		for (size_t l=m; l<=lmax_alm1; ++l) {
			almG(l,m)=resp1[l]*alm1(l,m);
			almC(l,m)=resp2[l]*alm1(l,m);
		}
	} 
	
	almZ.Set(lmax_alm1, lmax_alm1);
	alm2map_spin(job,almG,almZ,map1Q,map1U,1);
	alm2map_spin(job,almC,almZ,map3Q,map3U,3);
	
	#pragma omp parallel for
	for (int i=0; i< map1Q.Npix(); i++) {
		tempdouble=map1Q[i]*1.;
		map1Q[i]=map2Q[i]*(map1Q[i]-map3Q[i]) + map2U[i]*(map1U[i]-map3U[i]);
		map1U[i]=-map2Q[i]*(map1U[i]+map3U[i]) + map2U[i]*(tempdouble+map3Q[i]);
	}
  
	almG.Set(lmax_almG, lmax_almG);
	almC.Set(lmax_almG, lmax_almG);
	map2alm_spin_iter(job,map1Q,map1U,almG,almC,1,3);
	
	lsqr.alloc(lmax_almG+1);
	#pragma omp parallel for
	for (size_t l=2; l<=lmax_almG; l++) {
		lsqr[l]=sqrt(l*(l+1.))*.5;
	}
	
	almG.ScaleL(lsqr);
	almC.ScaleL(lsqr);
}

void lensEB(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, int nside, arr<double> &weight) {
	size_t lmax_alm1=alm1.Lmax();
	size_t lmax_alm2=alm2.Lmax();
	size_t lmax_almG=almG.Lmax();
	
	arr<double> resp1, resp2, lsqr;
	double tempdouble;

	almG.SetToZero();
	almC.SetToZero();
	
	Alm< xcomplex< double > > almZ(lmax_almG,lmax_almG);
	almZ.SetToZero();
	
	sharp_cxxjob<double> job;
	job.set_weighted_Healpix_geometry (nside, &weight[0]);
	
	Healpix_Map<double> map1Q,map1U,map2Q,map2U,map3Q,map3U;
	map1Q.SetNside(nside,RING);
	map1U.SetNside(nside,RING);
	map2Q.SetNside(nside,RING);
	map2U.SetNside(nside,RING);
	map3Q.SetNside(nside,RING);
	map3U.SetNside(nside,RING);
	
	almZ.Set(lmax_alm2, lmax_alm2);
    alm2map_spin(job,almZ,alm2,map2Q,map2U,2);
	
	resp1.alloc(lmax_alm1+1);
	resp2.alloc(lmax_alm1+1);
	#pragma omp parallel for
	for (size_t l=2; l<=lmax_alm1; l++) {
		resp1[l]=wcl.gg(l)*sqrt((l+2.)*(l-1.));
		resp2[l]=wcl.gg(l)*sqrt((l-2.)*(l+3.));
	}
	
	almG.Set(lmax_alm1, lmax_alm1);
	almC.Set(lmax_alm1, lmax_alm1);
	#pragma omp parallel for
	for (size_t m=0; m<=lmax_alm1; ++m) {
		for (size_t l=m; l<=lmax_alm1; ++l) {
			almG(l,m)=resp1[l]*alm1(l,m);
			almC(l,m)=resp2[l]*alm1(l,m);
		}
	} 
	
	almZ.Set(lmax_alm1, lmax_alm1);
	alm2map_spin(job,almG,almZ,map1Q,map1U,1);
	alm2map_spin(job,almC,almZ,map3Q,map3U,3);
	
	#pragma omp parallel for
	for (int i=0; i< map1Q.Npix(); i++) {
		tempdouble=map1Q[i]*1.;
		map1Q[i]=map2Q[i]*(map1Q[i]-map3Q[i]) + map2U[i]*(map1U[i]-map3U[i]);
		map1U[i]=-map2Q[i]*(map1U[i]+map3U[i]) + map2U[i]*(tempdouble+map3Q[i]);
	}
	
	almG.Set(lmax_almG, lmax_almG);
	almC.Set(lmax_almG, lmax_almG);
	map2alm_spin_iter(job,map1Q,map1U,almG,almC,1,3);
	
	lsqr.alloc(lmax_almG+1);
	#pragma omp parallel for
	for (size_t l=2; l<=lmax_almG; l++) {
		lsqr[l]=sqrt(l*(l+1.))*.5;
	}
	
	almG.ScaleL(lsqr);
	almC.ScaleL(lsqr);
}

void lensBB(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, int nside, arr<double> &weight) {
	size_t lmax_alm1=alm1.Lmax();
	size_t lmax_alm2=alm2.Lmax();
	size_t lmax_almG=almG.Lmax();
	
	arr<double> resp1, resp2, lsqr;
	double tempdouble;

	almG.SetToZero();
	almC.SetToZero();
	
	Alm< xcomplex< double > > almZ(lmax_almG,lmax_almG);
	almZ.SetToZero();
	
	sharp_cxxjob<double> job;
	job.set_weighted_Healpix_geometry (nside, &weight[0]);
	
	Healpix_Map<double> map1Q,map1U,map2Q,map2U,map3Q,map3U;
	map1Q.SetNside(nside,RING);
	map1U.SetNside(nside,RING);
	map2Q.SetNside(nside,RING);
	map2U.SetNside(nside,RING);
	map3Q.SetNside(nside,RING);
	map3U.SetNside(nside,RING);
	
	almZ.Set(lmax_alm2, lmax_alm2);
    alm2map_spin(job,alm2,almZ,map2Q,map2U,2);
	
	resp1.alloc(lmax_alm1+1);
	resp2.alloc(lmax_alm1+1);
	#pragma omp parallel for
	for (size_t l=2; l<=lmax_alm1; l++) {
		resp1[l]=wcl.cc(l)*sqrt((l+2.)*(l-1.));
		resp2[l]=wcl.cc(l)*sqrt((l-2.)*(l+3.));
	}
	
	almG.Set(lmax_alm1, lmax_alm1);
	almC.Set(lmax_alm1, lmax_alm1);
	#pragma omp parallel for
	for (size_t m=0; m<=lmax_alm1; ++m) {
		for (size_t l=m; l<=lmax_alm1; ++l) {
			almG(l,m)=resp1[l]*alm1(l,m);
			almC(l,m)=resp2[l]*alm1(l,m);
		}
	} 
	
	almZ.Set(lmax_alm1, lmax_alm1);
	alm2map_spin(job,almG,almZ,map1Q,map1U,1);
	alm2map_spin(job,almC,almZ,map3Q,map3U,3);
	
	#pragma omp parallel for
	for (int i=0; i< map1Q.Npix(); i++) {
		tempdouble=map1Q[i]*1.;
		map1Q[i]=map2Q[i]*(map1Q[i]-map3Q[i]) + map2U[i]*(map1U[i]-map3U[i]);
		map1U[i]=-map2Q[i]*(map1U[i]+map3U[i]) + map2U[i]*(tempdouble+map3Q[i]);
	}
  
	almG.Set(lmax_almG, lmax_almG);
	almC.Set(lmax_almG, lmax_almG);
	map2alm_spin_iter(job,map1Q,map1U,almG,almC,1,3);
	
	lsqr.alloc(lmax_almG+1);
	#pragma omp parallel for
	for (size_t l=2; l<=lmax_almG; l++) {
		lsqr[l]=sqrt(l*(l+1.))*.5;
	}
	
	almG.ScaleL(lsqr);
	almC.ScaleL(lsqr);
}