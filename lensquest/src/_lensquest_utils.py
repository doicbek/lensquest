import numpy as np

def cltype(cl):
	if type(cl)==np.ndarray:
		if cl.ndim==1:
			return 1
		else:
			return np.shape(cl)[0]
	elif type(cl)==tuple:
		return len(cl)
	else:
		return 0
		
def t2i(t):
	if t=="T": return 0
	elif t=="E": return 1
	elif t=="B": return 2