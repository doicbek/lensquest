import numpy as np

def cambdat2flat(A,lmax=None):
	ell=A[0]
	out=[]
	
	if lmax==None:
		lmax=len(ell)+1
	
	for cl in A[1:]:
		out.append(np.append([0,0],cl/ell/(ell+1)*2*np.pi)[:lmax+1])
		
	return out
	
def flat2cambdat(A,fname):
	if type(A)==np.ndarray: A=[A]
	
	l=np.arange(len(A[0]))
	
	out=[]
	
	for cl in A:
		out.append((l*(l+1)/2/np.pi*cl)[2:])
		
	np.savetxt(fname,out.T)

def loadcambdat(dat,lmax=None):
	return cambdat2flat(np.loadtxt(dat,unpack=1),lmax=lmax)
	
def get_cross(spectra,i,j):
	nspec=len(spectra)
	return spectra[min(i,j)]+spectra[max(i,j)]
	
def getweights(nl,spectra):
	lmax=len(list(nl.values())[0])
	l=np.arange(lmax)
	nmv=np.zeros(lmax)

	weight={}	
	for arg in spectra:
		weight[arg] = np.zeros(lmax)

	nspec=len(spectra)
	for L in range(2,lmax):
		mat=np.zeros((nspec,nspec))
		for i in range(nspec):
			for j in range(nspec):
				try:
					mat[i,j]=nl[get_cross(spectra,i,j)][L]#+self.n1[dh.get_cross(self.spectra,i,j)][L] #(1.+1./self.num_sims_MF)*
				except KeyError:
					mat[i,j]=0.0
		try:
			mat=np.linalg.inv(mat)
			nmv[L]=1./np.sum(mat)
		except:
			print('Matrix singular for L='+str(L))
			nmv[L]=0.
		for i in range(nspec):
			weight[spectra[i]][L]=nmv[L]*np.sum(mat[i,:])
			
	return nmv, weight
			
def whitenoise(lmax,w,theta,ellkneeT=0,alphaT=0,ellkneeP=0,alphaP=0,lmin=2):
	l=np.arange(lmax+1)
	npol=(w*np.pi/10800.)**2*np.ones(lmax+1)
	npol[:lmin]=np.zeros(lmin)
	pinkT=np.append(np.zeros(lmin),np.ones(lmax+1-lmin))
	if not (ellkneeT==0 and alphaT==0):
		pinkT[lmin:]+=(ellkneeT*1./l[lmin:])**alphaT
	pinkP=np.append(np.zeros(lmin),np.ones(lmax+1-lmin))
	if not (ellkneeP==0 and alphaP==0):
		pinkP[lmin:]+=(ellkneeP*1./l[lmin:])**alphaP
	beam=np.exp((l)*(l+1.)*(theta*np.pi/10800.)**2/8./np.log(2.))
	
	return npol,pinkT,pinkP,beam