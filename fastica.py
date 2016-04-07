import numpy as np
from scipy import linalg as cacca
import matplotlib.pyplot as plt

data = np.loadtxt("data.txt")
data = data.T # now it is (2x5000), N=2, M=5000

##############################
##		ORIGINAL DATA 		##
##############################

def plotData(data, string1, string2, label1="x1", label2="x2"):
	x1 = data[0,:]
	x2 = data[1,:]
	x = np.linspace(0, 100, len(x1))

	plt.figure(1)
	plt.scatter(x, x2, color='b', s=0.8)
	plt.title(string1)
	plt.xlabel("time")
	plt.ylabel(label1)
	plt.show()

	plt.figure(2)
	plt.scatter(x, x2, color='b', s=0.8)
	plt.title(string1)
	plt.xlabel("time")
	plt.ylabel(label2)
	plt.show()

	plt.figure(3)
	ax = plt.axes()
	plt.scatter(x1, x2, color='b', s=0.5)
	plt.title(string2)
	plt.xlabel(label1)
	plt.ylabel(label2)
	plt.show()

#plotData(data, "Mixed signals over time" ,"Mixed distribution")

##############################
##		PREPROCESSING		##
##############################

def QRalg(A):
	#initial step
	Q, R = cacca.qr(A)
	eigenVec = Q
	Ap = R.dot(Q)

	for i in range(10):
		Q, R = cacca.qr(Ap)
		Ap = R.dot(Q)
		eigenVec = eigenvec.dot(Q)
	
	eigenVal = np.diag(Ap)
	return eigenVal, eigenVec

#A = cacca.hessenberg(covar)
#eigenVal, eigenVec = QRalg(A)
#for i in range(10):
#	Q, R = cacca.qr(covar)
#	A = 
#sq = np.linalg.inv(cacca.sqrtm(covar))
#data = sq.dot(data)
#np.savetxt('readyData', data)
#print(np.cov(data))
#show plot

# centering the data
def centering(data):	
	#data = data - np.mean(data)
	for i in range(len(data)):
		data[i] = data[i] - np.mean(data[i])
	return data

# whitening
def whitening(data):
	covar = np.cov(data) # covMatrix is (2,2)
	#eigenVal, E = np.linalg.eig(covar)
	eigenVal, E = np.linalg.eigh(covar) #slightly different

	# another way to compute the inverse of a diagonal matrix
	#D = np.diagflat(1/eigenVal)

	D = np.linalg.inv(np.diagflat(eigenVal))
	D = cacca.sqrtm(D)
	#data = E.dot(D).dot(E.T).dot(data)
	data = D.dot(E.T).dot(data)
	
	return data, eigenVal, E

cent = centering(data)
transf, e, E = whitening(cent)

def plotEigen(data, E, e):
	plt.figure(1)
	ax = plt.axes()
	ax.arrow(0, 0, E[0,0]*e[0], E[1,0]*e[0], head_width=0.1, head_length=0.1, fc='r', ec='r')
	ax.arrow(0, 0, E[0,1]*e[1], E[1,1]*e[1], head_width=0.1, head_length=0.1, fc='r', ec='r')
	plt.scatter(data[0,:], data[1,:], color='b', s=0.5)
	plt.title("Eigenvectors on centered mixed distribution")
	plt.xlabel("x1")
	plt.ylabel("x2")
	plt.show()

#plotEigen(cent, E, e)
plotData(transf, "Whitened mixed signals over time", "Whitened mixed distribution")


##############################################
##		MULTIPLE COMPONENT EXTRACTION		##
##############################################

def f(u):
	return np.log(np.cosh(u))

def g(u):
	return np.tanh(u)

def g_p(u):
	return 1 - np.square(np.tanh(u))

def _gs_decorrelation(w, W, p):
	""" Gram-Schmidt-like decorrelation. """
	t = np.zeros_like(w)
	for j in range(p):
		t = t + np.dot(w, W[j]) * W[j]
		w -= t
	#w -= t
	return w


def extraction(X, w, maxit=100, tol=0.00001):
	# some parameters:
	N , M = X.shape
	C = len(w)
	I = np.ones(C)

	W = np.zeros((C,N)) # w->(C,N) = (2,2)

	for p in range(C):
		w0 = w[p]
		lim = tol+1 
		it = 0
		while ((lim > tol) & (it < maxit)):
			wtx = w0.T.dot(X)
			gwtx = g(wtx)
			g_wtx = g_p(wtx)

			w1 = (X * gwtx).mean(axis=1) - g_wtx.mean() * w0

			#decorrelation
			_gs_decorrelation(w1, W, p)
			w1 /= np.linalg.norm(w1)

			lim = (np.abs(np.abs(w1.T.dot(w0))-1))
			
			w0 = w1
			it += 1
		W[p,:] = w0
		print(it)
	return W
				
print(transf.shape)
w = np.random.randn(2,2)
s = extraction(transf, w).dot(transf)

#plotData(s, "Unmixed signals over time", "Unmixed 2d distribution", label1="s1", label2="s2")
