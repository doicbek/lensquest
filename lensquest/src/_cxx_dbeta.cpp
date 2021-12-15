void dbetaTT(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, std::vector<double>& rl1, std::vector<double>& rl2, int nside, arr<double> &weight) {
	size_t lmax_alm1=alm1.Lmax();
	size_t lmax_alm2=alm2.Lmax();
	size_t lmax_almG=almG.Lmax();
	
	arr<double> lsqr;
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
	
	almG.Set(lmax_alm1, lmax_alm1);
	#pragma omp parallel for
	for (size_t m=0; m<=lmax_alm2; ++m) {
		for (size_t l=m; l<=lmax_alm2; ++l) {
			almG(l,m)=rl1[l]*alm1(l,m);
		}
	} 
    alm2map_(job,almG,map1Q);
	
	almG.Set(lmax_alm2, lmax_alm2);
	#pragma omp parallel for
	for (size_t m=0; m<=lmax_alm2; ++m) {
		for (size_t l=m; l<=lmax_alm2; ++l) {
			almG(l,m)=wcl.tt(l)*alm2(l,m);
		}
	} 
	job.set_triangular_alm_info (lmax_alm2, lmax_alm2);	
	alm2map_(job,almG,map2Q);

	#pragma omp parallel for
	for (int i=0; i< map1Q.Npix(); i++) {
		map1Q[i]=map2Q[i]*map1Q[i];

	}

	almG.Set(lmax_almG, lmax_almG);
	map2alm_iter(map1Q, almG,3,weight);
}

void dbetaTE(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, std::vector<double>& rl1, std::vector<double>& rl2, int nside, arr<double> &weight) {
	size_t lmax_alm1=alm1.Lmax();
	size_t lmax_alm2=alm2.Lmax();
	size_t lmax_almG=almG.Lmax();
	
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
	
	almG.Set(lmax_alm2, lmax_alm2);
	#pragma omp parallel for
	for (size_t m=0; m<=lmax_alm1; ++m) {
		for (size_t l=m; l<=lmax_alm1; ++l) {
			almG(l,m)=rl2[l]*alm2(l,m);
		}
	} 
    almZ.Set(lmax_alm2, lmax_alm2);
    alm2map_spin(job,almG,almZ,map2Q,map2U,2);
	
	almG.Set(lmax_alm1, lmax_alm1);
	#pragma omp parallel for
	for (size_t m=0; m<=lmax_alm1; ++m) {
		for (size_t l=m; l<=lmax_alm1; ++l) {
			almG(l,m)=wcl.tg(l)*alm1(l,m);
		}
	} 
	almZ.Set(lmax_alm1, lmax_alm1);
	alm2map_spin(job,almG,almZ,map1Q,map1U,2);
	
	#pragma omp parallel for
	for (int i=0; i< map1Q.Npix(); i++) {
		map1Q[i]=map2Q[i]*map1Q[i] + map2U[i]*map1U[i];
	}
	
	almG.Set(lmax_almG, lmax_almG);
	map2alm_iter(map1Q, almG,3,weight);
}

void dbetaTB(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, std::vector<double>& rl1, std::vector<double>& rl2, int nside, arr<double> &weight) {
	size_t lmax_alm1=alm1.Lmax();
	size_t lmax_alm2=alm2.Lmax();
	size_t lmax_almG=almG.Lmax();
	
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
	
	almG.Set(lmax_alm2, lmax_alm2);
	#pragma omp parallel for
	for (size_t m=0; m<=lmax_alm1; ++m) {
		for (size_t l=m; l<=lmax_alm1; ++l) {
			almG(l,m)=rl2[l]*alm2(l,m);
		}
	} 
	almZ.Set(lmax_alm2, lmax_alm2);
    alm2map_spin(job,almZ,almG,map2Q,map2U,2);
	
	almG.Set(lmax_alm1, lmax_alm1);
	#pragma omp parallel for
	for (size_t m=0; m<=lmax_alm1; ++m) {
		for (size_t l=m; l<=lmax_alm1; ++l) {
			almG(l,m)=wcl.tg(l)*alm1(l,m);
		}
	} 
	almZ.Set(lmax_alm1, lmax_alm1);
	alm2map_spin(job,almG,almZ,map1Q,map1U,2);
	
	#pragma omp parallel for
	for (int i=0; i< map1Q.Npix(); i++) {
		map1Q[i]= map2Q[i]*map1Q[i] + map2U[i]*map1U[i];
	}
	
	almG.Set(lmax_almG, lmax_almG);
	map2alm_iter(map1Q, almG,3,weight);
}

void dbetaEE(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, std::vector<double>& rl1, std::vector<double>& rl2, int nside, arr<double> &weight) {
	size_t lmax_alm1=alm1.Lmax();
	size_t lmax_alm2=alm2.Lmax();
	size_t lmax_almG=almG.Lmax();
	
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
	
	
	almG.Set(lmax_alm2, lmax_alm2);
	#pragma omp parallel for
	for (size_t m=0; m<=lmax_alm1; ++m) {
		for (size_t l=m; l<=lmax_alm1; ++l) {
			almG(l,m)=rl2[l]*alm2(l,m);
		}
	} 
	almZ.Set(lmax_alm2, lmax_alm2);
    alm2map_spin(job,almG,almZ,map2Q,map2U,2);
	
	almG.Set(lmax_alm1, lmax_alm1);
	#pragma omp parallel for
	for (size_t m=0; m<=lmax_alm1; ++m) {
		for (size_t l=m; l<=lmax_alm1; ++l) {
			almG(l,m)=wcl.gg(l)*alm1(l,m);
		}
	} 
	almZ.Set(lmax_alm1, lmax_alm1);
	alm2map_spin(job,almG,almZ,map1Q,map1U,2);
	
	#pragma omp parallel for
	for (int i=0; i< map1Q.Npix(); i++) {
		map3[i]=map2Q[i]*map1Q[i] + map2U[i]*map1U[i];
	}

	almG.Set(lmax_alm2, lmax_alm2);
	#pragma omp parallel for
	for (size_t m=0; m<=lmax_alm2; ++m) {
		for (size_t l=m; l<=lmax_alm2; ++l) {
			almG(l,m)=rl1[l]*alm1(l,m);
		}
	} 
	almZ.Set(lmax_alm1, lmax_alm1);
    alm2map_spin(job,almG,almZ,map1Q,map1U,2);
	
	almG.Set(lmax_alm2, lmax_alm2);
	#pragma omp parallel for
	for (size_t m=0; m<=lmax_alm2; ++m) {
		for (size_t l=m; l<=lmax_alm2; ++l) {
			almG(l,m)=wcl.gg(l)*alm2(l,m);
		}
	} 
	almZ.Set(lmax_alm2, lmax_alm2);
	alm2map_spin(job,almG,almZ,map2Q,map2U,2);
	
	#pragma omp parallel for
	for (int i=0; i< map1Q.Npix(); i++) {
		map3[i]+=+map2Q[i]*map1Q[i] + map2U[i]*map1U[i];
	}

	
	almG.Set(lmax_almG, lmax_almG);
	map2alm_iter(map3, almG,3,weight);
	
	almG.Scale(.5);
}

void dbetaEB(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, std::vector<double>& rl1, std::vector<double>& rl2, int nside, arr<double> &weight) {
	size_t lmax_alm1=alm1.Lmax();
	size_t lmax_alm2=alm2.Lmax();
	size_t lmax_almG=almG.Lmax();
	
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
	
    almG.Set(lmax_alm2, lmax_alm2);
	#pragma omp parallel for
	for (size_t m=0; m<=lmax_alm1; ++m) {
		for (size_t l=m; l<=lmax_alm1; ++l) {
			almG(l,m)=rl2[l]*alm2(l,m);
		}
	} 
	almZ.Set(lmax_alm2, lmax_alm2);
    alm2map_spin(job,almZ,almG,map2Q,map2U,2);
	
	
	almG.Set(lmax_alm1, lmax_alm1);
	#pragma omp parallel for
	for (size_t m=0; m<=lmax_alm1; ++m) {
		for (size_t l=m; l<=lmax_alm1; ++l) {
			almG(l,m)=wcl.gg(l)*alm1(l,m);
		}
	} 
	almZ.Set(lmax_alm1, lmax_alm1);
	alm2map_spin(job,almG,almZ,map1Q,map1U,2);
	
	#pragma omp parallel for
	for (int i=0; i< map1Q.Npix(); i++) {
		map3[i]= + map2Q[i]*map1Q[i] + map2U[i]*map1U[i];
	}

	almG.Set(lmax_alm1, lmax_alm1);
	#pragma omp parallel for
	for (size_t m=0; m<=lmax_alm2; ++m) {
		for (size_t l=m; l<=lmax_alm2; ++l) {
			almG(l,m)=rl1[l]*alm1(l,m);
		}
	} 
    almZ.Set(lmax_alm1, lmax_alm1);
    alm2map_spin(job,almG,almZ,map1Q,map1U,2);
	
	almG.Set(lmax_alm2, lmax_alm2);
	#pragma omp parallel for
	for (size_t m=0; m<=lmax_alm2; ++m) {
		for (size_t l=m; l<=lmax_alm2; ++l) {
			almG(l,m)=wcl.cc(l)*alm2(l,m);
		}
	} 
	almZ.Set(lmax_alm2, lmax_alm2);
	alm2map_spin(job,almZ,almG,map2Q,map2U,2);
	
	#pragma omp parallel for
	for (int i=0; i< map1Q.Npix(); i++) {
		map3[i]+= + map2Q[i]*map1Q[i] + map2U[i]*map1U[i];
	}

	
	almG.Set(lmax_almG, lmax_almG);
	map2alm_iter(map3, almG,3,weight);
}


void dbetaBB(Alm< xcomplex< double > > & alm1, Alm< xcomplex< double > > & alm2, Alm< xcomplex< double > > & almG,  Alm< xcomplex< double > > & almC, PowSpec& wcl, std::vector<double>& rl1, std::vector<double>& rl2, int nside, arr<double> &weight) {
	size_t lmax_alm1=alm1.Lmax();
	size_t lmax_alm2=alm2.Lmax();
	size_t lmax_almG=almG.Lmax();
	
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
	
	almG.Set(lmax_alm2, lmax_alm2);
	#pragma omp parallel for
	for (size_t m=0; m<=lmax_alm1; ++m) {
		for (size_t l=m; l<=lmax_alm1; ++l) {
			almG(l,m)=rl2[l]*alm2(l,m);
		}
	} 
	almZ.Set(lmax_alm2, lmax_alm2);
    alm2map_spin(job,almG,almZ,map2Q,map2U,2);

	almG.Set(lmax_alm1, lmax_alm1);
	#pragma omp parallel for
	for (size_t m=0; m<=lmax_alm1; ++m) {
		for (size_t l=m; l<=lmax_alm1; ++l) {
			almG(l,m)=wcl.cc(l)*alm1(l,m);
		}
	} 
	almZ.Set(lmax_alm1, lmax_alm1);
	alm2map_spin(job,almG,almZ,map1Q,map1U,2);
	
	#pragma omp parallel for
	for (int i=0; i< map1Q.Npix(); i++) {
		map3[i]=map2Q[i]*map1Q[i] + map2U[i]*map1U[i];
	}

	almG.Set(lmax_alm1, lmax_alm1);
	#pragma omp parallel for
	for (size_t m=0; m<=lmax_alm2; ++m) {
		for (size_t l=m; l<=lmax_alm2; ++l) {
			almG(l,m)=rl1[l]*alm1(l,m);
		}
	} 
	almZ.Set(lmax_alm1, lmax_alm1);
    alm2map_spin(job,almG,almZ,map1Q,map1U,2);
	

	almG.Set(lmax_alm2, lmax_alm2);
	#pragma omp parallel for
	for (size_t m=0; m<=lmax_alm2; ++m) {
		for (size_t l=m; l<=lmax_alm2; ++l) {
			almG(l,m)=wcl.cc(l)*alm2(l,m);
		}
	} 
	almZ.Set(lmax_alm2, lmax_alm2);
	alm2map_spin(job,almG,almZ,map2Q,map2U,2);
	
	#pragma omp parallel for
	for (int i=0; i< map1Q.Npix(); i++) {
		map3[i]+=+map2Q[i]*map1Q[i] + map2U[i]*map1U[i];
	}

	
	almG.Set(lmax_almG, lmax_almG);
	map2alm_iter(map3, almG,3,weight);
	
	almG.Scale(.5);
}
