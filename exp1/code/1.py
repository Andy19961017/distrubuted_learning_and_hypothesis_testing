import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import random as rd
import math

node_interested=1

def parse_parameters():
	file1 = open('../csv/'+sys.argv[1]+'.csv' ,"r")
	data1 = csv.reader(file1 , delimiter= ",")
	file2 = open('../csv/'+sys.argv[2]+'.csv' ,"r")
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

def mylog(b):
	if b <= 1.0e-216:
		return math.log(1.0e-216)
	else:
		return math.log(b)

def get_observation():
	for i in range(num_of_node):
		X[i]=0 if rd.random()<F[true_theta-1+i*num_of_theta,0] else 1

def observe(node):
	get_observation()
	global estimaiton_history, b, q
	estimaiton_history=np.concatenate((estimaiton_history,q[node-1,:].reshape(1,-1)),axis=0)
	for i in range(num_of_node):
		s=0
		for theta in range(num_of_theta):
			# print('s',i,'+=',F[i*num_of_theta+theta,X[i]],'*',q[i,theta])
			s+=F[i*num_of_theta+theta,X[i]]*q[i,theta]
		for theta in range(num_of_theta):
			# print('b',i,theta,'=',F[i*num_of_theta+theta,X[i]],'*',q[i,theta],'/',s)
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

def show_history(node):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	fig.suptitle(sys.argv[1]+'_'+sys.argv[2]+'_'+sys.argv[3]+'_'+sys.argv[4],
		fontsize = 14, fontweight='bold')
	ax.set_xlabel("Number of iterations, t")
	ax.set_ylabel("Estimation for node "+str(node))
	for i in range(num_of_theta):
		ax.plot(range(iteration),estimaiton_history[:,i])
	l=()
	for i in range(num_of_theta):
		l+=('theta '+str(i+1),)
	plt.legend(l)
	plt.grid(True)
	plt.show()

if len(sys.argv) != 5:
	print("Usage: python3 1.py W F iteration true_theta")
	'''for example  python3 1.py W1 F1 100 1'''
	sys.exit(0)
true_theta=int(sys.argv[4])
W, F, num_of_node, num_of_theta =parse_parameters()
print(W,'\n',F)
X=np.zeros((num_of_node)).astype(int)
estimaiton_history=np.zeros((0,num_of_theta))
iteration=int(sys.argv[3])
q=np.full((num_of_node, num_of_theta), 1.0/num_of_theta)
b=np.full((num_of_node, num_of_theta), 0.)
for i in range(iteration):
	# print("iteration",i)
	observe(node_interested)  #The node interested
show_history(node_interested) #The node interested

# print(w)
# print(v)