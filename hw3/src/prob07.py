import sys
import math

#import numpy
#from numpy import matrix
#from numpy import linalg
#from numpy import array


def read_feature(data_file_name):
	"""
	modified from svmutil.svm_read_problem()

	svm_read_problem(data_file_name) -> [y, x]
	Read LIBSVM-format data from data_file_name and return labels y
	and data instances x.
	"""
	prob_y = []
	prob_x = []
	for line in open(data_file_name):
		#print line
		line = line.rsplit(None, 1)
		#print line
		# In case an instance with all zero features
		#if len(line) == 1: line += ['']
		features, label = line 
		#print features
		#print label
		#parse prob_x
		xi = []
		ind = 1
		for e in features.split():
			xi.append(float(e))
			ind += 1
		#parse prob_y
		prob_y += [float(label)]
		prob_x += [xi]	
	return (prob_y, prob_x)

def sign(x):
	if x >= 0:
		return +1
	return -1
def not_equal(x, y):
	if x == y:
		return 0
	return 1
# Data : zip of x (data[i][0]), y (data[i][1]), u (data[i][2])
def e_u_in(data, i, s, theta):
	err = 0.0
	for j in range(len(data)):
		err += data[j][2] * not_equal(data[j][1], s * sign(data[j][0][i] - theta))
	return err


def decision_stump(data):
	S = [-1, 1]
	neg_inf = float("-inf")
	s_opt, i_opt, theta_opt = (0, 0, 0)
	ein_opt = float("inf")
	for i in range(len(data[0][0])):
		data = sorted(data, key=lambda x: x[0][i])
		#print data
		for j in range(len(data)):
			for s in S:
				if j == 0:
					theta = neg_inf
				else:
					theta = 0.5*(data[j][0][i] + data[j-1][0][i])

				ein = e_u_in(data, i, s, theta)
				#print (s, i, theta, ein), (s_opt, i_opt, theta_opt, ein_opt) 
				if  ein < ein_opt:
					s_opt = s
					i_opt = i
					theta_opt = theta
					ein_opt = ein
	#print (s_opt, i_opt, theta_opt)
	return (s_opt, i_opt, theta_opt)

#stump = (s, i, theta)
# for et_g, et_G, no need to zip u (weight)
def et_g(data, stump):
	err = 0
	for i in range(len(data)):
		err += not_equal(data[i][1], stump[0] * sign(data[i][0][stump[1]] - stump[2]))
	return float(err) / float(len(data))

def et_G(data, stumpl, alphal):
	err = 0
	for i in range(len(data)):
		G = 0
		val = 0.0
		for j in range(len(stumpl)):
			val += alphal[j] * ( stumpl[j][0] * sign(data[i][0][stumpl[j][1]] - stumpl[j][2]) )
		if val >= 0.0:
			G = +1
		else:
			G = -1
		err += not_equal(data[i][1], G)
	return float(err) / float(len(data))

def test_decision_error(x, y, stump):
	if y == stump[0] * sign(x[stump[1]]-stump[2]):
		return False
	return True


def update_u(data, stump, epsilon):
	new_u = []
	update_factor = math.sqrt((1.0 - epsilon) / epsilon)
	for i in range(len(data)):
		if test_decision_error(data[i][0], data[i][1], stump) == True:
			new_u.append(data[i][2] * update_factor)
		else:
			new_u.append(data[i][2] / update_factor)
	return new_u

def weight_sum(weight):
	sum_u = 0.0
	for i in range(len(weight)):
		sum_u += weight[i]
	return sum_u

def epsilon(data, stump):
	sum_u = 0.0
	sum_incorrect_u = 0.0
	for i in range(len(data)):
		sum_u += data[i][2]
		if test_decision_error(data[i][0], data[i][1], stump) == True:
			sum_incorrect_u += data[i][2]
	return sum_incorrect_u / sum_u

def alpha(epsilon):
	return math.log(math.sqrt((1.0 - epsilon) / epsilon) )

y, x = read_feature('data/hw3_train.dat')

testy, testx = read_feature('data/hw3_test.dat')

u = [1.0 / len(x)]*len(x)

#print x
#print y
#print u

#decision_stump(zip(x, y, u))

T = 300
Alpha = []
Stump = []

# for statistics
Et_in = []
Et_out = []
Et_in_agg = []
Et_out_agg = []
Epsilon = []
Ut = []


for t in range(T):
	current_stump = decision_stump(zip(x, y, u))
	current_epsilon = epsilon(zip(x, y, u), current_stump)
	# stat
	Et_in.append(et_g(zip(x, y), current_stump ))
	Et_out.append(et_g(zip(testx, testy), current_stump ))
	Ut.append(weight_sum(u))

	#print "iteration = %d: u = "%(t), u

	u = update_u(zip(x, y, u), current_stump, current_epsilon)
	Stump.append(current_stump)
	Alpha.append(alpha(current_epsilon))

	#stat
	Epsilon.append(current_epsilon)
	Et_in_agg.append(et_G(zip(x, y), Stump, Alpha))
	Et_out_agg.append(et_G(zip(testx, testy), Stump, Alpha))





print "=============== Raw data of Q07 ==============="
print "Ein(g) = ", Et_in
print "=============== Raw data of Q09 ==============="
print "Ein(G) = ", Et_in_agg
print "=============== Raw data of Q10 ==============="
print "Ut = ", Ut
print "=============== Raw data of Q11 ==============="
print "epsilon =", Epsilon
print "=============== Raw data of Q12 ==============="
print "Eout(g) = ", Et_out
print "=============== Raw data of Q13 ==============="
print "Eout(G) = ", Et_out_agg
print "=============== End of Raw Data ==============="
print "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
print "=============== Results of Q07  ==============="
print "Ein(g1) = %f, alpha1 = %f"%(Et_in[0], Alpha[0])
print "=============== Results of Q09  ==============="
print "Ein(G) = %f"%(Et_in_agg[T-1])
print "=============== Results of Q10  ==============="
print "U_2 = %f, U_T = %f"%(Ut[1], Ut[T-1])
print "=============== Results of Q11  ==============="
print "min(epsilon) = %f"%(min(Epsilon))
print "=============== Results of Q12  ==============="
print "Eout(g1) = %f"%(Et_out[0])
print "=============== Results of Q13  ==============="
print "Eout(G) = %f"%(Et_out_agg[T-1])
print "==============================================="