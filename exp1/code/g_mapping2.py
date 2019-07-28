import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import random as rd
import math
from math import log
from time import time

node_interested=1

def compute_rate(a11, a12, a21, a22, b11, b12, b21, b22):
	def f11(x):
		return a11 if x==1 else 1-a11
	def f12(x):
		return a12 if x==1 else 1-a12
	def f21(x):
		return a21 if x==1 else 1-a21
	def f22(x):
		return a22 if x==1 else 1-a22
	def g11(x):
		return b11 if x==1 else 1-b11
	def g12(x):
		return b12 if x==1 else 1-b12
	def g21(x):
		return b21 if x==1 else 1-b21
	def g22(x):
		return b22 if x==1 else 1-b22
	def get_expected_value(prob, numerator, denominator):
		return prob(0)*log(numerator(0)/denominator(0))+prob(1)*log(numerator(1)/denominator(1))
	
	D_f21=0.5*get_expected_value(f12, f12, f11)+0.5*get_expected_value(f22, f22, f21)
	D_f12=0.5*get_expected_value(f11, f11, f12)+0.5*get_expected_value(f21, f21, f22)
	D_f=min(D_f21, D_f12)

	D_g21=0.5*get_expected_value(f12, g12, g11)+0.5*get_expected_value(f22, g22, g21)
	D_g12=0.5*get_expected_value(f11, g11, g12)+0.5*get_expected_value(f21, g21, g22)
	D_g=min(D_g21, D_g12)

	# print("D_f21, D_f12, D_g21, D_g12", D_f21, D_f12, D_g21, D_g12)

	return D_g21 if true_theta==2 else D_g12

def mylog(b):
	if b <= 1.0e-216:
		return math.log(1.0e-216)
	else:
		return math.log(b)

def get_observation(F):
	X=np.zeros((num_of_node)).astype(int)
	for i in range(num_of_node):
		X[i]=0 if rd.random()<F[true_theta-1+i*num_of_theta,0] else 1
	return X

def observe(node,F,g,estimaiton_history,q,b):
	X=get_observation(F)
	estimaiton_history=np.concatenate((estimaiton_history,q[node-1,:].reshape(1,-1)),axis=0)
	for i in range(num_of_node):
		s=0
		for theta in range(num_of_theta):
			s+=g[i*num_of_theta+theta,X[i]]*q[i,theta]
		for theta in range(num_of_theta):
			b[i,theta]=g[i*num_of_theta+theta,X[i]]*q[i,theta]/s
	for i in range(num_of_node):
		s=0
		for theta in range(num_of_theta):
			ss=0
			for j in range(num_of_node):
				ss+=W[i,j]*mylog(b[j,theta])
			s+=math.exp(ss)
		for theta in range(num_of_theta):
			sss=0
			for j in range(num_of_node):
				sss+=W[i,j]*mylog(b[j,theta])
			q[i,theta]=math.exp(sss)/s
	return estimaiton_history,q,b

def plot_history(fig, ind, error, r, para):
	ax = fig.add_subplot(3,2,2*ind-1)
	fig.suptitle('num_of_trails: '+str(num_of_trail), fontsize = 14)
		# fontsize = 14, fontweight='bold')
	# ax.title.set_text('rate:'+str(r)+'\n'+str(para))
	ax.set_title('rate:'+str(r), fontdict={'fontsize':14})
	ax.plot(range(iteration),error)
	plt.grid(True)
	ax = fig.add_subplot(3,2,2*ind)
	ax.set_title('rate:'+str(r), fontdict={'fontsize':14})
	# ax.title.set_text('rate:'+str(r)+'\n'+str(para), fontsize = 8)
	ax.plot(range(iteration),np.log(error))
	plt.grid(True)

def get_special_case(a11, a12, a21, a22, yy1, yy2, k):
	y1=yy1*k
	y2=yy2*k
	x1=(a11+a12-2)/(a11+a12)*y1
	b11=(1-math.exp(y1))/(math.exp(x1)-math.exp(y1))
	b12=(1-math.exp(-y1))/(math.exp(-x1)-math.exp(-y1))
	x2=(a21+a22-2)/(a21+a22)*y2
	b21=(1-math.exp(y2))/(math.exp(x2)-math.exp(y2))
	b22=(1-math.exp(-y2))/(math.exp(-x2)-math.exp(-y2))
	return b11, b12, b21, b22

def test(para, fig, ind):
	(a11, a12, a21, a22, b11, b12, b21, b22)=para
	r=compute_rate(a11, a12, a21, a22, b11, b12, b21, b22)
	F=np.array([[1-a11,a11],[1-a12,a12],[1-a21,a21],[1-a22,a22]])
	g=np.array([[1-b11,b11],[1-b12,b12],[1-b21,b21],[1-b22,b22]])
	estimaiton_history=np.zeros((0,num_of_theta))
	q=np.full((num_of_node, num_of_theta), 1.0/num_of_theta)
	b=np.full((num_of_node, num_of_theta), 0.)
	for i in range(iteration):
		estimaiton_history,q,b=observe(node_interested,F,g,estimaiton_history,q,b)  #The node interested
	error=[(1.0 if estimaiton_history[t,false_theta-1]>estimaiton_history[t,true_theta-1] else 0.0) for t in range(estimaiton_history.shape[0])]
	return error, r


if len(sys.argv) != 3:
	print("Usage: python3 g_mapping2.py iteration true_theta")
	sys.exit(0)
true_theta=int(sys.argv[2])
# theta_interested=2 if true_theta==1 else 1
false_theta=2 if true_theta==1 else 1

W=np.array([[0.5,0.5],[0.5,0.5]])
num_of_node, num_of_theta = 2, 2
iteration=int(sys.argv[1])
num_of_trail=1000

# a11, a12, a21, a22 = 0.2, 0.25, 0.6, 0.65
a11, a12, a21, a22 = 0.2, 0.4, 0.8, 0.6
# a11, a12, a21, a22 = 0.3, 0.8, 0.7, 0.6
y1=-1 if a11<a12 else 1
y2=-1 if a21<a22 else 1

fig = plt.figure()

error1, error2, error3 = np.zeros((iteration)), np.zeros((iteration)), np.zeros((iteration))

para1=(a11, a12, a21, a22, a11, a12, a21, a22)
print(para1)
for i in range(num_of_trail):
	e,r1=test(para1, fig, 1)
	error1=error1+e
error1=error1/num_of_trail
plot_history(fig, 1, error1, r1, para1)

b11, b12, b21, b22 = get_special_case(a11, a12, a21, a22, y1, y2, 0.1)
# para2=(a11, a12, a21, a22, b11, b12, b21, b22)
para2=(a11, a12, a21, a22, 0.89, 0.9, 0.11, 0.1)
print(para2)
for i in range(num_of_trail):
	e,r2=test(para2, fig, 2)
	error2=error2+e
error2=error2/num_of_trail
plot_history(fig, 2, error2, r2, para2)

b11, b12, b21, b22 = get_special_case(a11, a12, a21, a22, y1, y2, 0.7)
para3=(a11, a12, a21, a22, b11, b12, b21, b22)
print(para3)
for i in range(num_of_trail):
	e,r3=test(para3, fig, 3)
	error3=error3+e
error3=error3/num_of_trail
plot_history(fig, 3, error3, r3, para3)

fig.tight_layout()
t=str(time())[-4:]
plt.savefig(t)
with open(t+'.log', 'w') as f:
	f.write(str(para1)+'\n')
	f.write(str(para2)+'\n')
	f.write(str(para3)+'\n')
plt.show()






