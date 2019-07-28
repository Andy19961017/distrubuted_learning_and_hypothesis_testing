import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import random as rd
import math

node_interested=5
theta_interested=3

def parse_parameters(f):
	file1 = open('../csv/'+sys.argv[1]+'.csv' ,"r")
	data1 = csv.reader(file1 , delimiter= ",")
	file2 = open('../csv/'+f+'.csv' ,"r")
	data2 = csv.reader(file2 , delimiter= ",")
	# read weight
	W=[]
	row_num=0
	for row in data1:
		# print(row)
		if row_num != 0:
			r=[]
			for i in range(len(row)):
				r.append(row[i])
			W.append(r)
		row_num+=1
	W.pop()
	W=np.array(W).astype(float)
	num_of_node=W.shape[0]
	# print()

	# read PDF
	F=[]
	row_num=0
	for row in data2:
		# print(row)
		if row_num != 0 and row_num != 1 and row_num != 2:
			r=[]
			for i in range(1,len(row)):
				r.append(row[i])
			F.append(r)
		row_num+=1
	F.pop()
	F=np.array(F).astype(float)
	num_of_theta=F.shape[0]//num_of_node

	file1.close()
	file2.close()

	return W, F, num_of_node, num_of_theta

def get_observation():
	for i in range(num_of_node):
		X[i]=0 if rd.random()<F[true_theta-1+i*num_of_theta,0] else 1

def mylog(b):
	if b <= 1.0e-216:
		return math.log(1.0e-216)
	else:
		return math.log(b)

def observe(node):
	get_observation()
	global estimaiton_history, b, q
	estimaiton_history=np.concatenate((estimaiton_history,q[node-1,:].reshape(1,-1)),axis=0)
	for i in range(num_of_node):
		s=0
		for theta in range(num_of_theta):
			s+=F[i*num_of_theta+theta,X[i]]*q[i,theta]
		for theta in range(num_of_theta):
			b[i,theta]=F[i*num_of_theta+theta,X[i]]*q[i,theta]/s
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

def show_history(node,theta_num):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	subtitle=sys.argv[1]
	for x in sys.argv[2:]:
		subtitle+=('_'+x)
	fig.suptitle(subtitle, fontsize = 14, fontweight='bold')
	ax.set_xlabel("Number of iterations, t")
	ax.set_ylabel("Log estimation of theta "+str(theta_num)+" of node "+str(node))
	for his in estimaiton_history_list:
		ax.plot(range(iteration),np.log(his[:,theta_num-1]))
	for tK in t_K_list:
		ax.plot(range(iteration),tK)
	l=()
	for f in sys.argv[4:]:
		l+=(f,)
	for f in sys.argv[4:]:
		l+=('K('+sys.argv[3]+','+str(theta_interested)+') for '+f,)
	plt.legend(l)
	plt.grid(True)
	plt.show()

def KLDiv(node,true_t,intr_t):
	div=0
	for i in range(2):
		div-=F[node*num_of_theta+true_t][i]*math.log(F[node*num_of_theta+intr_t][i]/F[node*num_of_theta+true_t][i])
	return div

def compute_K():
	ss=0
	w,v=np.linalg.eig(W.T)
	for i in range(num_of_node):
		if 0.9999<w[i] and w[i]<1.0001:
			ss=v[:,i]
	ss/=np.sum(ss)
	K=0
	for i in range(num_of_node):
		K+=ss[i]*KLDiv(i,int(sys.argv[3])-1,theta_interested-1)
	return K

#main
if len(sys.argv) < 5:
	print("Usage: python3 2.py W iteration true_theta F1 F2 ...")
	'''for example python3 2.py W3 100 1 F2 F3'''
	sys.exit(0)
true_theta=int(sys.argv[3])
estimaiton_history_list=[]
t_K_list=[]
for f in sys.argv[4:]:
	iteration=int(sys.argv[2])
	W, F, num_of_node, num_of_theta =parse_parameters(f)
	K=compute_K()
	# print(K)
	t_K=[]
	for t in range(iteration):
		t_K.append((-1)*t*K)
	t_K_list.append(t_K)
	X=np.zeros((num_of_node)).astype(int)
	estimaiton_history=np.zeros((0,num_of_theta))
	q=np.full((num_of_node, num_of_theta), 1.0/num_of_theta)
	b=np.full((num_of_node, num_of_theta), 0.)
	for t in range(iteration):
		observe(node_interested)
	estimaiton_history_list.append(estimaiton_history)
show_history(node_interested,theta_interested)