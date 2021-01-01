#ifndef KERNELS_CPP
#define KERNELS_CPP

#include <omp.h>
#include <vector>
#include <math.h>

using namespace std;

void compF(vector<vector<double> >& F, int l1, int spin, int lmax_l2, int lmax_l3){

	vector<double> wigner_output;
	int wigner_lmax;
	int l2_max;
	int l2_min;
	
	double srt2l1=sqrt((2.0*l1+1.0)/16.0/M_PI);
	
	#pragma omp parallel private(l2_max, l2_min, wigner_lmax, wigner_output)       
	{
		#pragma omp for
		for (int l3=0;l3<lmax_l2;l3++){
			
			double srt2l3=sqrt(2.0*l3+1.0);
			
			for (int l2=0;l2<lmax_l3;l2++) {
				F[l2][l3]=0.0;
			}
			
			if (l3>=spin) {
				wigner_output=WignerSymbols::wigner3j(l1,l3,0,spin,-spin);
				
				l2_min=max(abs(l1-l3),0);
				l2_max=l1+l3;
				
				if (l2_max+1>=lmax_l2) {wigner_lmax=lmax_l2;}
				else {wigner_lmax=l2_max+1;}
				
				for (int l2=l2_min;l2<wigner_lmax;l2++) {
					F[l2][l3]=wigner_output[l2-l2_min];
				}
			} 

			for (int l2=0;l2<lmax_l2;l2++) {
				F[l2][l3]*=(-l1*(l1+1.0)+l2*(l2+1.0)+l3*(l3+1.0))*sqrt(2.0*l2+1.0)*srt2l1*srt2l3;
			}
		}
	}
}

void compF_phi(vector<vector<double> >& F, int l2, int spin, int lmax){

	vector<double> wigner_output;
	int wigner_lmax;
	int l3_max;
	int l3_min;
	
	double sqt2l2=sqrt((2.0*l2+1.0)/16.0/M_PI);
	
	#pragma omp parallel private(l3_max, l3_min, wigner_lmax, wigner_output)       
	{
		#pragma omp for
		for (int l1=0;l1<lmax;l1++){
			
			double srt2l1=sqrt(2.0*l1+1.0);
			
			for (int l3=0;l3<lmax;l3++) {
				F[l1][l3]=0.0;
			}
			
			if (l1>=spin) {
				wigner_output=WignerSymbols::wigner3j(l1,l2,-spin,spin,0);
				
				l3_min=max(abs(l2-l1),spin);
				l3_max=l2+l1;
				
				if (l3_max+1>=lmax) {wigner_lmax=lmax;}
				else {wigner_lmax=l3_max+1;}
				
				for (int l3=l3_min;l3<wigner_lmax;l3++) {
					F[l1][l3]=wigner_output[l3-l3_min]; 
				}
			} 

			for (int l3=0;l3<lmax;l3++) {
				F[l1][l3]*=(-l1*(l1+1.0)+l2*(l2+1.0)+l3*(l3+1.0));
				F[l1][l3]*=sqt2l2;
				F[l1][l3]*=srt2l1;
				F[l1][l3]*=sqrt(2.0*l3+1.0);
			}
		}
	}
}


void compF_curl(vector<vector<double> >& F, int l2, int spin, int lmax){

	vector<double> wigner_output_a;
	vector<double> wigner_output_b;
	
	vector<double> prefactor_a(lmax, 0.0);
	vector<double> prefactor_b(lmax, 0.0);
	
	#pragma omp parallel for
	for (int l3=spin;l3<lmax;l3++) {
		prefactor_a[l3]=sqrt((l3+1.0-spin)*(l3+spin)*(2.0*l3+1.0));
		prefactor_b[l3]=sqrt((l3-spin)*(l3+1.0+spin)*(2.0*l3+1.0));
	}

	int wigner_lmax;
	int l3_max;
	int l3_min_a;
	int l3_min_b;
	
	double sqt2l2=sqrt((2.0*l2+1.0)*l2*(l2+1.0)/16.0/M_PI);
	
	#pragma omp parallel private(l3_max, l3_min_a, l3_min_b, wigner_lmax, wigner_output_a, wigner_output_b)       
	{
		#pragma omp for
		for (int l1=0;l1<lmax;l1++){
			
			double sqt2l1=sqrt(2.0*l1+1.0);
			
			for (int l3=0;l3<lmax;l3++) {
				F[l1][l3]=0.0;
			}
			
			if (l1>=spin) {
				wigner_output_a=WignerSymbols::wigner3j(l1,l2,1-spin,spin,-1);
				wigner_output_b=WignerSymbols::wigner3j(l1,l2,-1-spin,spin,1);
				
				l3_min_a=max(abs(l2-l1),abs(1-spin));
				l3_min_b=max(abs(l2-l1),abs(-1-spin));
				l3_max=l2+l1;
				
				if (l3_max+1>=lmax) {wigner_lmax=lmax;}
				else {wigner_lmax=l3_max+1;}
				
				for (int l3=l3_min_a;l3<wigner_lmax;l3++) {
					F[l1][l3]=prefactor_a[l3]*wigner_output_a[l3-l3_min_a];
				}
				for (int l3=l3_min_b;l3<wigner_lmax;l3++) {
					F[l1][l3]+=-prefactor_b[l3]*wigner_output_b[l3-l3_min_b];
				}
			}

			for (int l3=0;l3<lmax;l3++) {
				F[l1][l3]*=sqt2l2;
				F[l1][l3]*=sqt2l1;
			}
		}
	}
}

void compF_a(vector<vector<double> >& F, int l2, int lmax){

	vector<double> wigner_output;
	int wigner_lmax;
	int l3_max;
	int l3_min;
	
	double sqt2l2=sqrt((2.0*l2+1.0)/4.0/M_PI);
	
	#pragma omp parallel private(l3_max, l3_min, wigner_lmax, wigner_output)       
	{
		#pragma omp for
		for (int l1=0;l1<lmax;l1++){
			
			double srt2l1=sqrt(2.0*l1+1.0);
			
			for (int l3=0;l3<lmax;l3++) {
				F[l1][l3]=0.0;
			}
			
			if (l1>=2) {
				wigner_output=WignerSymbols::wigner3j(l1,l2,-2,2,0);
				
				l3_min=max(abs(l2-l1),2);
				l3_max=l2+l1;
				
				if (l3_max+1>=lmax) {wigner_lmax=lmax;}
				else {wigner_lmax=l3_max+1;}
				
				for (int l3=l3_min;l3<wigner_lmax;l3++) {
					F[l1][l3]=wigner_output[l3-l3_min]; 
				}
			} 

			for (int l3=0;l3<lmax;l3++) {
				F[l1][l3]*=sqt2l2;
				F[l1][l3]*=srt2l1;
				F[l1][l3]*=sqrt(2.0*l3+1.0);
			}
		}
	}
}

void compF_f(vector<vector<double> >& F, int l2, int lmax){

	vector<double> wigner_output;
	int wigner_lmax;
	int l3_max;
	int l3_min;
	
	double sqt2l2=sqrt((2.0*l2+1.0)/4.0/M_PI);
	
	#pragma omp parallel private(l3_max, l3_min, wigner_lmax, wigner_output)       
	{
		#pragma omp for
		for (int l1=0;l1<lmax;l1++){
			
			double srt2l1=sqrt(2.0*l1+1.0);
			
			for (int l3=0;l3<lmax;l3++) {
				F[l1][l3]=0.0;
			}
			
			if (l1>=2) {
				wigner_output=WignerSymbols::wigner3j(l1,l2,2,2,-4);
				
				l3_min=max(abs(l2-l1),2);
				l3_max=l2+l1;
				
				if (l3_max+1>=lmax) {wigner_lmax=lmax;}
				else {wigner_lmax=l3_max+1;}
				
				for (int l3=l3_min;l3<wigner_lmax;l3++) {
					F[l1][l3]=wigner_output[l3-l3_min]; 
				}
			} 

			for (int l3=0;l3<lmax;l3++) {
				F[l1][l3]*=sqt2l2;
				F[l1][l3]*=srt2l1;
				F[l1][l3]*=sqrt(2.0*l3+1.0);
			}
		}
	}
}

void compF_gamma(vector<vector<double> >& F, int l2, int lmax){

	vector<double> wigner_output;
	int wigner_lmax;
	int l3_max;
	int l3_min;
	
	double sqt2l2=sqrt((2.0*l2+1.0)/4.0/M_PI);
	
	#pragma omp parallel private(l3_max, l3_min, wigner_lmax, wigner_output)       
	{
		#pragma omp for
		for (int l1=0;l1<lmax;l1++){
			
			double srt2l1=sqrt(2.0*l1+1.0);
			
			for (int l3=0;l3<lmax;l3++) {
				F[l1][l3]=0.0;
			}
			
			if (l1>=2) {
				wigner_output=WignerSymbols::wigner3j(l1,l2,0,2,-2);
				
				l3_min=max(abs(l2-l1),0);
				l3_max=l2+l1;
				
				if (l3_max+1>=lmax) {wigner_lmax=lmax;}
				else {wigner_lmax=l3_max+1;}
				
				for (int l3=l3_min;l3<wigner_lmax;l3++) {
					F[l1][l3]=wigner_output[l3-l3_min]; 
				}
			} 

			for (int l3=0;l3<lmax;l3++) {
				F[l1][l3]*=sqt2l2;
				F[l1][l3]*=srt2l1;
				F[l1][l3]*=sqrt(2.0*l3+1.0);
			}
		}
	}
}

void compF_pa(vector<vector<double> >& F, int l2, int lmax){

	vector<double> wigner_output;
	int wigner_lmax;
	int l3_max;
	int l3_min;
	
	double sqt2l2=sqrt((2.0*l2+1.0)/4.0/M_PI);
	
	#pragma omp parallel private(l3_max, l3_min, wigner_lmax, wigner_output)       
	{
		#pragma omp for
		for (int l1=0;l1<lmax;l1++){
			
			double srt2l1=sqrt(2.0*l1+1.0);
			
			for (int l3=0;l3<lmax;l3++) {
				F[l1][l3]=0.0;
			}
			
			if (l1>=2) {
				wigner_output=WignerSymbols::wigner3j(l1,l2,3,-2,-1);
				
				l3_min=max(abs(l2-l1),3);
				l3_max=l2+l1;
				
				if (l3_max+1>=lmax) {wigner_lmax=lmax;}
				else {wigner_lmax=l3_max+1;}
				
				for (int l3=l3_min;l3<wigner_lmax;l3++) {
					F[l1][l3]=wigner_output[l3-l3_min]; 
				}
			} 

			for (int l3=0;l3<lmax;l3++) {
				F[l1][l3]*=sqt2l2;
				F[l1][l3]*=srt2l1;
				F[l1][l3]*=sqrt((2.0*l3+1.0)*(l3-2.0)*(l3+3.0));
			}
		}
	}
}

void compF_pb(vector<vector<double> >& F, int l2, int lmax){

	vector<double> wigner_output;
	int wigner_lmax;
	int l3_max;
	int l3_min;
	
	double sqt2l2=sqrt((2.0*l2+1.0)/4.0/M_PI);
	
	#pragma omp parallel private(l3_max, l3_min, wigner_lmax, wigner_output)
	{
		#pragma omp for
		for (int l1=0;l1<lmax;l1++){
			
			double srt2l1=sqrt(2.0*l1+1.0);
			
			for (int l3=0;l3<lmax;l3++) {
				F[l1][l3]=0.0;
			}
			
			if (l1>=2) {
				wigner_output=WignerSymbols::wigner3j(l1,l2,-1,2,-1);
				
				l3_min=max(abs(l2-l1),1);
				l3_max=l2+l1;
				
				if (l3_max+1>=lmax) {wigner_lmax=lmax;}
				else {wigner_lmax=l3_max+1;}
				
				for (int l3=l3_min;l3<wigner_lmax;l3++) {
					F[l1][l3]=wigner_output[l3-l3_min]; 
				}
			} 

			for (int l3=0;l3<lmax;l3++) {
				F[l1][l3]*=sqt2l2;
				F[l1][l3]*=srt2l1;
				F[l1][l3]*=sqrt((2.0*l3+1.0)*(l3+2.0)*(l3-1.0));
			}
		}
	}
}

void compF_d(vector<vector<double> >& F, int l2, int lmax){

	vector<double> wigner_output;
	int wigner_lmax;
	int l3_max;
	int l3_min;
	
	double sqt2l2=sqrt((2.0*l2+1.0)/4.0/M_PI);
	
	#pragma omp parallel private(l3_max, l3_min, wigner_lmax, wigner_output)
	{
		#pragma omp for
		for (int l1=0;l1<lmax;l1++){
			
			double srt2l1=sqrt(2.0*l1+1.0);
			
			for (int l3=0;l3<lmax;l3++) {
				F[l1][l3]=0.0;
			}
			
			if (l1>=2) {
				wigner_output=WignerSymbols::wigner3j(l1,l2,-1,2,-1);
				
				l3_min=max(abs(l2-l1),1);
				l3_max=l2+l1;
				
				if (l3_max+1>=lmax) {wigner_lmax=lmax;}
				else {wigner_lmax=l3_max+1;}
				
				for (int l3=l3_min;l3<wigner_lmax;l3++) {
					F[l1][l3]=wigner_output[l3-l3_min]; 
				}
			} 

			for (int l3=0;l3<lmax;l3++) {
				F[l1][l3]*=sqt2l2;
				F[l1][l3]*=srt2l1;
				F[l1][l3]*=sqrt((2.0*l3+1.0)*l3*(l3+1.0)/2.0);
			}
		}
	}
}

void compF_q(vector<vector<double> >& F, int l2, int lmax){

	vector<double> wigner_output;
	int wigner_lmax;
	int l3_max;
	int l3_min;
	
	double sqt2l2=sqrt((2.0*l2+1.0)/4.0/M_PI);
	
	#pragma omp parallel private(l3_max, l3_min, wigner_lmax, wigner_output)
	{
		#pragma omp for
		for (int l1=0;l1<lmax;l1++){
			
			double srt2l1=sqrt(2.0*l1+1.0);
			
			for (int l3=0;l3<lmax;l3++) {
				F[l1][l3]=0.0;
			}
			
			if (l1>=2) {
				wigner_output=WignerSymbols::wigner3j(l1,l2,-2,2,0);
				
				l3_min=max(abs(l2-l1),2);
				l3_max=l2+l1;
				
				if (l3_max+1>=lmax) {wigner_lmax=lmax;}
				else {wigner_lmax=l3_max+1;}
				
				for (int l3=l3_min;l3<wigner_lmax;l3++) {
					F[l1][l3]=wigner_output[l3-l3_min]; 
				}
			} 

			for (int l3=0;l3<lmax;l3++) {
				F[l1][l3]*=sqt2l2;
				F[l1][l3]*=srt2l1;
				F[l1][l3]*=sqrt((2.0*l3+1.0)*(l3-1.0)*l3*(l3+1.0)*(l3+2.0))/2.0;
			}
		}
	}
}



void compF_2_noise(vector<vector<double> >& F, int l2, int spin, int lmax){

	vector<double> wigner_output;
	int wigner_lmax;
	int l3_max;
	int l3_min;
	
	double sqt2l2=sqrt((2.0*l2+1.0)/4.0/M_PI);
	
	#pragma omp parallel private(l3_max, l3_min, wigner_lmax, wigner_output)       
	{
		#pragma omp for
		for (int l1=0;l1<lmax;l1++){
			
			double srt2l1=sqrt(2.0*l1+1.0);
			
			for (int l3=0;l3<lmax;l3++) {
				F[l1][l3]=0.0;
			}
			
			wigner_output=WignerSymbols::wigner3j(l1,l2,-spin,spin,0);
			
			l3_min=abs(l2-l1);
			l3_max=l2+l1;
			
			if (l3_max+1>=lmax) {wigner_lmax=lmax;}
			else {wigner_lmax=l3_max+1;}
			
			for (int l3=l3_min;l3<wigner_lmax;l3++) {
				F[l1][l3]=wigner_output[l3-l3_min]*sqt2l2*srt2l1*sqrt(2.0*l3+1.0); 
			}
		}
	}
}

void compF_2_mask(vector<vector<double> >& F, int l2, int spin, int lmax){

	vector<double> wigner_output;
	int wigner_lmax;
	int l3_max;
	int l3_min;

	double sqt2l2=(2.0*l2+1.0)/4.0/M_PI;
	
	sqt2l2=sqrt(sqt2l2);
	
	#pragma omp parallel private(l3_max, l3_min, wigner_lmax, wigner_output)       
	{
		#pragma omp for
		for (int l1=0;l1<lmax;l1++){
			
			double srt2l1=sqrt(2.0*l1+1.0);
			
			for (int l3=0;l3<lmax;l3++) {
				F[l1][l3]=0.0;
			}
			
			if (l1>=spin) {
				wigner_output=WignerSymbols::wigner3j(l1,l2,-spin,spin,0);
				
				l3_min=max(abs(l2-l1),spin);
				l3_max=l2+l1;
				
				if (l3_max+1>=lmax) {wigner_lmax=lmax;}
				else {wigner_lmax=l3_max+1;}
				
				for (int l3=l3_min;l3<wigner_lmax;l3++) {
					F[l1][l3]=wigner_output[l3-l3_min]; 
				}
			} 

			for (int l3=0;l3<lmax;l3++) {
				F[l1][l3]*=sqt2l2;
				F[l1][l3]*=srt2l1;
				F[l1][l3]*=sqrt(2.0*l3+1.0);
			}
		}
	}
}

void compF_6j(vector<double>& F, int l2, int l3, int l4, int l5, int l6, int lmax){

	vector<double> wigner_output;
	int wigner_lmax;
	int l1_max;
	int l1_min;
	
	#pragma omp parallel for
	for (int l1=0;l1<lmax;l1++) {
		F[l1]=0.0;
	}
	
	l1_min=max(abs(l2-l3),abs(l5-l6));
	l1_max=min(abs(l2+l3),abs(l5+l6));
	
	bool select = (
		abs(l4-l2) <= l6 && l6 <= l4+l2
		&& abs(l4-l5) <= l3 && l3 <= l4+l5
		);
		
	select = (select && l1_max+1-l1_min>0);
	
	if (select) {
		wigner_output=WignerSymbols::wigner6j(l2, l3, l4, l5, l6);
	
		if (l1_max+1>=lmax) {wigner_lmax=lmax;}
		else {wigner_lmax=l1_max+1;}
	
		#pragma omp parallel for
		for (int l1=l1_min;l1<wigner_lmax;l1++) {
			F[l1]=wigner_output[l1-l1_min]; 
			if (F[l1]>1e20) throw bad_exception();
		}
	}
}

// void compF_2_m(vector<vector<double> >& F, int l2, int M, int m1, int m2, int lmax){

	// vector<double> wigner_output;
	// int wigner_lmax;
	// int l1_max;
	// int l1_min;
		
	// #pragma omp parallel private(l1_max, l1_min, wigner_lmax, wigner_output)       
	// {
		// #pragma omp for
		// for (int l3=0;l3<lmax;l3++){
						
			// for (int l1=0;l1<lmax;l1++) {
				// F[l1][l3]=0.0;
			// }
			
			// if (l3>=abs(m2)) {
				// wigner_output=WignerSymbols::wigner3j(l3,l2,m1,m2,-M);
				
				// l1_min=max(abs(l2-l3),abs(m1));
				// l1_max=l2+l3;
				
				// if (l1_max+1>=lmax) {wigner_lmax=lmax;}
				// else {wigner_lmax=l1_max+1;}
				
				// for (int l1=l1_min;l1<wigner_lmax;l1++) {
					// F[l1][l3]=wigner_output[l1-l1_min]; 
				// }
			// } 
		// }
	// }
// }

#endif /* KERNELS_CPP */
