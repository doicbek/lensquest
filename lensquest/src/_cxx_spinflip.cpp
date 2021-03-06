void spiTE(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, int nside, arr<double> &weight) {
	size_t lmax_alm1=alm1.Lmax();
	size_t lmax_alm2=alm2.Lmax();
	size_t lmax_almG=almG.Lmax();
	
	arr<double> resp1;
	double tempdouble;

	almG.SetToZero();
	almC.SetToZero();
	
	Alm< xcomplex< double > > almZ(lmax_almG,lmax_almG);
	almZ.SetToZero();
	
	sharp_cxxjob<double> job;
	job.set_weighted_Healpix_geometry (nside, &weight[0]);
	
	Healpix_Map<double> map1Q,map1U,map2Q,map2U,map3;
	map1Q.SetNside(nside,RING);
	map1U.SetNside(nside,RING);
	map2Q.SetNside(nside,RING);
	map2U.SetNside(nside,RING);
	map3.SetNside(nside,RING);

	almZ.Set(lmax_alm2, lmax_alm2);
    alm2map_spin(job,alm2,almZ,map2Q,map2U,2);
	
	resp1.alloc(lmax_alm1+1);
	#pragma omp parallel for
	for (size_t l=2; l<=lmax_alm1; l++) {
		resp1[l]=wcl.tg(l);
	}
	
	almG.Set(lmax_alm1, lmax_alm1);
	#pragma omp parallel for
	for (size_t m=0; m<=lmax_alm1; ++m) {
		for (size_t l=m; l<=lmax_alm1; ++l) {
			almG(l,m)=resp1[l]*alm1(l,m);
		}
	} 
	
	almZ.Set(lmax_alm1, lmax_alm1);
	alm2map_spin(job,almG,almZ,map1Q,map1U,2);
	
	#pragma omp parallel for
	for (int i=0; i< map1Q.Npix(); i++) {
		tempdouble=map1Q[i];
		map1Q[i]=map2Q[i]*tempdouble - map2U[i]*map1U[i];
		map1U[i]=-map2U[i]*tempdouble - map2Q[i]*map1U[i];
	}
	
	almG.Set(lmax_almG, lmax_almG);
	almC.Set(lmax_almG, lmax_almG);
	map2alm_spin_iter(job,map1Q,map1U,almG,almC,4,3);
}

void spiTB(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, int nside, arr<double> &weight) {
	size_t lmax_alm1=alm1.Lmax();
	size_t lmax_alm2=alm2.Lmax();
	size_t lmax_almG=almG.Lmax();
	
	arr<double> resp1;
	double tempdouble;

	almG.SetToZero();
	almC.SetToZero();
	
	Alm< xcomplex< double > > almZ(lmax_almG,lmax_almG);
	almZ.SetToZero();
	
	sharp_cxxjob<double> job;
	job.set_weighted_Healpix_geometry (nside, &weight[0]);
	
	Healpix_Map<double> map1Q,map1U,map2Q,map2U;
	map1Q.SetNside(nside,RING);
	map1U.SetNside(nside,RING);
	map2Q.SetNside(nside,RING);
	map2U.SetNside(nside,RING);
	
	almZ.Set(lmax_alm2, lmax_alm2);
    alm2map_spin(job,almZ,alm2,map2Q,map2U,2);
	
	resp1.alloc(lmax_alm1+1);
	#pragma omp parallel for
	for (size_t l=2; l<=lmax_alm1; l++) {
		resp1[l]=wcl.tg(l);
	}
	
	almG.Set(lmax_alm1, lmax_alm1);
	#pragma omp parallel for
	for (size_t m=0; m<=lmax_alm1; ++m) {
		for (size_t l=m; l<=lmax_alm1; ++l) {
			almG(l,m)=resp1[l]*alm1(l,m);
		}
	} 
	
	almZ.Set(lmax_alm1, lmax_alm1);
	alm2map_spin(job,almG,almZ,map1Q,map1U,2);
	
	#pragma omp parallel for
	for (int i=0; i< map1Q.Npix(); i++) {
		tempdouble=map1Q[i];
		map1Q[i]=map2Q[i]*tempdouble - map2U[i]*map1U[i];
		map1U[i]=-map2U[i]*tempdouble - map2Q[i]*map1U[i];
	}
	
	almG.Set(lmax_almG, lmax_almG);
	almC.Set(lmax_almG, lmax_almG);
	map2alm_spin_iter(job,map1Q,map1U,almG,almC,4,3);
}

void spiEE(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, int nside, arr<double> &weight) {
	size_t lmax_alm1=alm1.Lmax();
	size_t lmax_alm2=alm2.Lmax();
	size_t lmax_almG=almG.Lmax();
	
	arr<double> resp1;
	double tempdouble;

	almG.SetToZero();
	almC.SetToZero();
	
	Alm< xcomplex< double > > almZ(lmax_almG,lmax_almG);
	almZ.SetToZero();
	
	sharp_cxxjob<double> job;
	job.set_weighted_Healpix_geometry (nside, &weight[0]);
	
	Healpix_Map<double> map1Q,map1U,map2Q,map2U,map3,map4;
	map1Q.SetNside(nside,RING);
	map1U.SetNside(nside,RING);
	map2Q.SetNside(nside,RING);
	map2U.SetNside(nside,RING);
	map3.SetNside(nside,RING);
	map4.SetNside(nside,RING);
	
	almZ.Set(lmax_alm2, lmax_alm2);
    alm2map_spin(job,alm2,almZ,map2Q,map2U,2);
	
	resp1.alloc(lmax_alm1+1);
	#pragma omp parallel for
	for (size_t l=2; l<=lmax_alm1; l++) {
		resp1[l]=wcl.gg(l);
	}
	
	almG.Set(lmax_alm1, lmax_alm1);
	#pragma omp parallel for
	for (size_t m=0; m<=lmax_alm1; ++m) {
		for (size_t l=m; l<=lmax_alm1; ++l) {
			almG(l,m)=resp1[l]*alm1(l,m);
		}
	} 
	
	almZ.Set(lmax_alm1, lmax_alm1);
	alm2map_spin(job,almG,almZ,map1Q,map1U,2);
	
	#pragma omp parallel for
	for (int i=0; i< map1Q.Npix(); i++) {
		map3[i]=map2Q[i]*map1Q[i] - map2U[i]*map1U[i];
		map4[i]=-map2U[i]*map1Q[i] - map2Q[i]*map1U[i];
	}

	almZ.Set(lmax_alm1, lmax_alm1);
    alm2map_spin(job,alm1,almZ,map1Q,map1U,2);
	
	resp1.alloc(lmax_alm2+1);
	#pragma omp parallel for
	for (size_t l=2; l<=lmax_alm1; l++) {
		resp1[l]=wcl.gg(l);
	}
	
	almG.Set(lmax_alm2, lmax_alm2);
	#pragma omp parallel for
	for (size_t m=0; m<=lmax_alm2; ++m) {
		for (size_t l=m; l<=lmax_alm2; ++l) {
			almG(l,m)=resp1[l]*alm2(l,m);
		}
	} 
	
	almZ.Set(lmax_alm2, lmax_alm2);
	alm2map_spin(job,almG,almZ,map2Q,map2U,2);
		
	#pragma omp parallel for
	for (int i=0; i< map1Q.Npix(); i++) {
		tempdouble=map1Q[i];
		map1Q[i]=map3[i]+map2Q[i]*tempdouble - map2U[i]*map1U[i];
		map1U[i]=map4[i]-map2U[i]*tempdouble - map2Q[i]*map1U[i];
	}
	
	almG.Set(lmax_almG, lmax_almG);
	almC.Set(lmax_almG, lmax_almG);
	map2alm_spin_iter(job,map1Q,map1U,almG,almC,4,3);

	almG.Scale(.5);
	almC.Scale(.5);
}

void spiEB(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, int nside, arr<double> &weight) {
	size_t lmax_alm1=alm1.Lmax();
	size_t lmax_alm2=alm2.Lmax();
	size_t lmax_almG=almG.Lmax();
	
	arr<double> resp1;
	double tempdouble;

	almG.SetToZero();
	almC.SetToZero();
	
	Alm< xcomplex< double > > almZ(lmax_almG,lmax_almG);
	almZ.SetToZero();
	
	sharp_cxxjob<double> job;
	job.set_weighted_Healpix_geometry (nside, &weight[0]);
	
	Healpix_Map<double> map1Q,map1U,map2Q,map2U,map3,map4;
	map1Q.SetNside(nside,RING);
	map1U.SetNside(nside,RING);
	map2Q.SetNside(nside,RING);
	map2U.SetNside(nside,RING);
	map3.SetNside(nside,RING);
	map4.SetNside(nside,RING);

	almZ.Set(lmax_alm2, lmax_alm2);
    alm2map_spin(job,almZ,alm2,map2Q,map2U,2);
	
	resp1.alloc(lmax_alm1+1);
	#pragma omp parallel for
	for (size_t l=2; l<=lmax_alm1; l++) {
		resp1[l]=wcl.gg(l);
	}
	
	almG.Set(lmax_alm1, lmax_alm1);
	#pragma omp parallel for
	for (size_t m=0; m<=lmax_alm1; ++m) {
		for (size_t l=m; l<=lmax_alm1; ++l) {
			almG(l,m)=resp1[l]*alm1(l,m);
		}
	} 
	
	almZ.Set(lmax_alm1, lmax_alm1);
	alm2map_spin(job,almG,almZ,map1Q,map1U,2);
	
	#pragma omp parallel for
	for (int i=0; i< map1Q.Npix(); i++) {
		map3[i]=map2Q[i]*map1Q[i] - map2U[i]*map1U[i];
		map4[i]=-map2U[i]*map1Q[i] - map2Q[i]*map1U[i];
	}

	almZ.Set(lmax_alm1, lmax_alm1);
    alm2map_spin(job,alm1,almZ,map1Q,map1U,2);
	
	resp1.alloc(lmax_alm2+1);
	#pragma omp parallel for
	for (size_t l=2; l<=lmax_alm1; l++) {
		resp1[l]=wcl.cc(l);
	}
	
	almG.Set(lmax_alm2, lmax_alm2);
	#pragma omp parallel for
	for (size_t m=0; m<=lmax_alm2; ++m) {
		for (size_t l=m; l<=lmax_alm2; ++l) {
			almG(l,m)=resp1[l]*alm2(l,m);
		}
	} 
	
	almZ.Set(lmax_alm2, lmax_alm2);
	alm2map_spin(job,almZ,almG,map2Q,map2U,2);
	
	#pragma omp parallel for
	for (int i=0; i< map1Q.Npix(); i++) {
		tempdouble=map1Q[i];
		map1Q[i]=map3[i]+map2Q[i]*tempdouble - map2U[i]*map1U[i];
		map1U[i]=map4[i]-map2U[i]*tempdouble - map2Q[i]*map1U[i];
	}

	almG.Set(lmax_almG, lmax_almG);
	almC.Set(lmax_almG, lmax_almG);
	map2alm_spin_iter(job,map1Q,map1U,almG,almC,4,3);
}