void dirTE(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, int nside, arr<double> &weight) {
	size_t lmax_alm1=alm1.Lmax();
	size_t lmax_alm2=alm2.Lmax();
	size_t lmax_almG=almG.Lmax();
	
	arr<double> resp1,resp2;

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
	resp2.alloc(lmax_alm1+1);
	#pragma omp parallel for
	for (size_t l=2; l<=lmax_alm1; l++) {
		resp1[l]=sqrt((l-2.)*(l+3.))*wcl.tg(l);
		resp2[l]=sqrt((l+2.)*(l-1.))*wcl.tg(l);
	}
	
	almG.Set(lmax_alm1, lmax_alm1);
	#pragma omp parallel for
	for (size_t m=0; m<=lmax_alm1; ++m) {
		for (size_t l=m; l<=lmax_alm1; ++l) {
			almG(l,m)=resp1[l]*alm1(l,m);
		}
	} 
	
	almZ.Set(lmax_alm1, lmax_alm1);
	alm2map_spin(job,almG,almZ,map1Q,map1U,3);
	
	#pragma omp parallel for
	for (int i=0; i< map1Q.Npix(); i++) {
		map3[i]= - map2Q[i]*map1U[i] + map2U[i]*map1Q[i];
		map4[i]= map2Q[i]*map1Q[i] + map2U[i]*map1U[i];
	}
	
	almG.Set(lmax_alm1, lmax_alm1);
	#pragma omp parallel for
	for (size_t m=0; m<=lmax_alm1; ++m) {
		for (size_t l=m; l<=lmax_alm1; ++l) {
			almG(l,m)=resp2[l]*alm1(l,m);
		}
	} 
	
	almZ.Set(lmax_alm1, lmax_alm1);
	alm2map_spin(job,almG,almZ,map1Q,map1U,1);
	
	#pragma omp parallel for
	for (int i=0; i< map1Q.Npix(); i++) {
		map3[i]+= - map2Q[i]*map1U[i] + map2U[i]*map1Q[i];
		map4[i]+= - map2Q[i]*map1Q[i] - map2U[i]*map1U[i];
	}
	
	almG.Set(lmax_almG, lmax_almG);
	almC.Set(lmax_almG, lmax_almG);
	map2alm_spin_iter(job,map3,map4,almC,almG,1,3);	
}

void dirTB(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, int nside, arr<double> &weight) {
	size_t lmax_alm1=alm1.Lmax();
	size_t lmax_alm2=alm2.Lmax();
	size_t lmax_almG=almG.Lmax();
	
	arr<double> resp1,resp2;

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
	resp2.alloc(lmax_alm1+1);
	#pragma omp parallel for
	for (size_t l=2; l<=lmax_alm1; l++) {
		resp1[l]=sqrt((l-2.)*(l+3.))*wcl.tg(l);
		resp2[l]=sqrt((l+2.)*(l-1.))*wcl.tg(l);
	}
	
	almG.Set(lmax_alm1, lmax_alm1);
	#pragma omp parallel for
	for (size_t m=0; m<=lmax_alm1; ++m) {
		for (size_t l=m; l<=lmax_alm1; ++l) {
			almG(l,m)=resp1[l]*alm1(l,m);
		}
	} 
	
	almZ.Set(lmax_alm1, lmax_alm1);
	alm2map_spin(job,almG,almZ,map1Q,map1U,3);
	
	#pragma omp parallel for
	for (int i=0; i< map1Q.Npix(); i++) {
		map3[i]= + map2Q[i]*map1U[i] - map2U[i]*map1Q[i];
		map4[i]= - map2Q[i]*map1Q[i] - map2U[i]*map1U[i];
	}
	
	almG.Set(lmax_alm1, lmax_alm1);
	#pragma omp parallel for
	for (size_t m=0; m<=lmax_alm1; ++m) {
		for (size_t l=m; l<=lmax_alm1; ++l) {
			almG(l,m)=resp2[l]*alm1(l,m);
		}
	} 
	
	almZ.Set(lmax_alm1, lmax_alm1);
	alm2map_spin(job,almG,almZ,map1Q,map1U,1);
	
	#pragma omp parallel for
	for (int i=0; i< map1Q.Npix(); i++) {
		map3[i]+= - map2Q[i]*map1U[i] + map2U[i]*map1Q[i];
		map4[i]+= - map2Q[i]*map1Q[i] - map2U[i]*map1U[i];
	}
	
	almG.Set(lmax_almG, lmax_almG);
	almC.Set(lmax_almG, lmax_almG);
	map2alm_spin_iter(job,map3,map4,almC,almG,1,3);	
}

void dirEE(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, int nside, arr<double> &weight) {
	size_t lmax_alm1=alm1.Lmax();
	size_t lmax_alm2=alm2.Lmax();
	size_t lmax_almG=almG.Lmax();
	
	arr<double> resp1,resp2;

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
	resp2.alloc(lmax_alm1+1);
	#pragma omp parallel for
	for (size_t l=2; l<=lmax_alm1; l++) {
		resp1[l]=sqrt((l-2.)*(l+3.))*wcl.gg(l);
		resp2[l]=sqrt((l+2.)*(l-1.))*wcl.gg(l);
	}
	
	almG.Set(lmax_alm1, lmax_alm1);
	#pragma omp parallel for
	for (size_t m=0; m<=lmax_alm1; ++m) {
		for (size_t l=m; l<=lmax_alm1; ++l) {
			almG(l,m)=resp1[l]*alm1(l,m);
		}
	} 
	
	almZ.Set(lmax_alm1, lmax_alm1);
	alm2map_spin(job,almG,almZ,map1Q,map1U,3);
	
	#pragma omp parallel for
	for (int i=0; i< map1Q.Npix(); i++) {
		map3[i]= - map2Q[i]*map1U[i] + map2U[i]*map1Q[i];
		map4[i]= + map2Q[i]*map1Q[i] + map2U[i]*map1U[i];
	}
	
	almG.Set(lmax_alm1, lmax_alm1);
	#pragma omp parallel for
	for (size_t m=0; m<=lmax_alm1; ++m) {
		for (size_t l=m; l<=lmax_alm1; ++l) {
			almG(l,m)=resp2[l]*alm1(l,m);
		}
	} 
	
	almZ.Set(lmax_alm1, lmax_alm1);
	alm2map_spin(job,almG,almZ,map1Q,map1U,1);
	
	#pragma omp parallel for
	for (int i=0; i< map1Q.Npix(); i++) {
		map3[i]+= + map2Q[i]*map1U[i] - map2U[i]*map1Q[i];
		map4[i]+= + map2Q[i]*map1Q[i] + map2U[i]*map1U[i];
	}
	
	almG.Set(lmax_almG, lmax_almG);
	almC.Set(lmax_almG, lmax_almG);
	map2alm_spin_iter(job,map3,map4,almC,almG,1,3);	

	almZ.Set(lmax_alm2, lmax_alm2);
    alm2map_spin(job,alm1,almZ,map2Q,map2U,2);
	
	resp1.alloc(lmax_alm1+1);
	resp2.alloc(lmax_alm1+1);
	#pragma omp parallel for
	for (size_t l=2; l<=lmax_alm1; l++) {
		resp1[l]=sqrt((l-2.)*(l+3.))*wcl.gg(l);
		resp2[l]=sqrt((l+2.)*(l-1.))*wcl.gg(l);
	}
	
	almG.Set(lmax_alm1, lmax_alm1);
	#pragma omp parallel for
	for (size_t m=0; m<=lmax_alm1; ++m) {
		for (size_t l=m; l<=lmax_alm1; ++l) {
			almG(l,m)=resp1[l]*alm2(l,m);
		}
	} 
	
	almZ.Set(lmax_alm1, lmax_alm1);
	alm2map_spin(job,almG,almZ,map1Q,map1U,3);
	
	#pragma omp parallel for
	for (int i=0; i< map1Q.Npix(); i++) {
		map3[i]+= + map2Q[i]*map1U[i] - map2U[i]*map1Q[i];
		map4[i]+= + map2Q[i]*map1Q[i] + map2U[i]*map1U[i];
	}
	
	almG.Set(lmax_alm1, lmax_alm1);
	#pragma omp parallel for
	for (size_t m=0; m<=lmax_alm1; ++m) {
		for (size_t l=m; l<=lmax_alm1; ++l) {
			almG(l,m)=resp2[l]*alm1(l,m);
		}
	} 
	
	almZ.Set(lmax_alm1, lmax_alm1);
	alm2map_spin(job,almG,almZ,map1Q,map1U,1);
	
	#pragma omp parallel for
	for (int i=0; i< map1Q.Npix(); i++) {
		map3[i]+= - map2Q[i]*map1U[i] + map2U[i]*map1Q[i];
		map4[i]+= + map2Q[i]*map1Q[i] + map2U[i]*map1U[i];
	}
	
	almG.Set(lmax_almG, lmax_almG);
	almC.Set(lmax_almG, lmax_almG);
	map2alm_spin_iter(job,map3,map4,almC,almG,1,3);	
}

void dirEB(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, int nside, arr<double> &weight) {
	size_t lmax_alm1=alm1.Lmax();
	size_t lmax_alm2=alm2.Lmax();
	size_t lmax_almG=almG.Lmax();
	
	arr<double> resp1,resp2;

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
	resp2.alloc(lmax_alm1+1);
	#pragma omp parallel for
	for (size_t l=2; l<=lmax_alm1; l++) {
		resp1[l]=sqrt((l-2.)*(l+3.))*wcl.gg(l);
		resp2[l]=sqrt((l+2.)*(l-1.))*wcl.gg(l);
	}
	
	almG.Set(lmax_alm1, lmax_alm1);
	#pragma omp parallel for
	for (size_t m=0; m<=lmax_alm1; ++m) {
		for (size_t l=m; l<=lmax_alm1; ++l) {
			almG(l,m)=resp1[l]*alm1(l,m);
		}
	} 
	
	almZ.Set(lmax_alm1, lmax_alm1);
	alm2map_spin(job,almG,almZ,map1Q,map1U,3);
	
	#pragma omp parallel for
	for (int i=0; i< map1Q.Npix(); i++) {
		map3[i]= + map2Q[i]*map1U[i] - map2U[i]*map1Q[i];
		map4[i]= - map2Q[i]*map1Q[i] - map2U[i]*map1U[i];
	}
	
	almG.Set(lmax_alm1, lmax_alm1);
	#pragma omp parallel for
	for (size_t m=0; m<=lmax_alm1; ++m) {
		for (size_t l=m; l<=lmax_alm1; ++l) {
			almG(l,m)=resp2[l]*alm1(l,m);
		}
	} 
	
	almZ.Set(lmax_alm1, lmax_alm1);
	alm2map_spin(job,almG,almZ,map1Q,map1U,1);
	
	#pragma omp parallel for
	for (int i=0; i< map1Q.Npix(); i++) {
		map3[i]+= - map2Q[i]*map1U[i] + map2U[i]*map1Q[i];
		map4[i]+= - map2Q[i]*map1Q[i] - map2U[i]*map1U[i];
	}
	
	almG.Set(lmax_almG, lmax_almG);
	almC.Set(lmax_almG, lmax_almG);
	map2alm_spin_iter(job,map3,map4,almC,almG,1,3);	

	almZ.Set(lmax_alm2, lmax_alm2);
    alm2map_spin(job,alm1,almZ,map2Q,map2U,2);
	
	resp1.alloc(lmax_alm1+1);
	resp2.alloc(lmax_alm1+1);
	#pragma omp parallel for
	for (size_t l=2; l<=lmax_alm1; l++) {
		resp1[l]=sqrt((l-2.)*(l+3.))*wcl.cc(l);
		resp2[l]=sqrt((l+2.)*(l-1.))*wcl.cc(l);
	}
	
	almG.Set(lmax_alm1, lmax_alm1);
	#pragma omp parallel for
	for (size_t m=0; m<=lmax_alm1; ++m) {
		for (size_t l=m; l<=lmax_alm1; ++l) {
			almG(l,m)=resp1[l]*alm2(l,m);
		}
	} 
	
	almZ.Set(lmax_alm1, lmax_alm1);
	alm2map_spin(job,almZ,almG,map1Q,map1U,3);
	
	#pragma omp parallel for
	for (int i=0; i< map1Q.Npix(); i++) {
		map3[i]+= - map2Q[i]*map1U[i] + map2U[i]*map1Q[i];
		map4[i]+= - map2Q[i]*map1Q[i] - map2U[i]*map1U[i];
	}
	
	almG.Set(lmax_alm1, lmax_alm1);
	#pragma omp parallel for
	for (size_t m=0; m<=lmax_alm1; ++m) {
		for (size_t l=m; l<=lmax_alm1; ++l) {
			almG(l,m)=resp2[l]*alm1(l,m);
		}
	} 
	
	almZ.Set(lmax_alm1, lmax_alm1);
	alm2map_spin(job,almZ,almG,map1Q,map1U,1);
	
	#pragma omp parallel for
	for (int i=0; i< map1Q.Npix(); i++) {
		map3[i]+= + map2Q[i]*map1U[i] - map2U[i]*map1Q[i];
		map4[i]+= - map2Q[i]*map1Q[i] - map2U[i]*map1U[i];
	}
	
	almG.Set(lmax_almG, lmax_almG);
	almC.Set(lmax_almG, lmax_almG);
	map2alm_spin_iter(job,map3,map4,almC,almG,1,3);	
}