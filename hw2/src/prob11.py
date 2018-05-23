import sys
import math

import numpy
from numpy import matrix
from numpy import linalg
from numpy import array


def svm_read_feature(data_file_name):
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

def gaussian_kernel(x_1, x_2, gamma):
	return math.exp(-gamma * numpy.inner(array(x_1) - array(x_2), array(x_1) - array(x_2)) )

def construct_unit_matrix(size):
	array = []

	for i in range(size):
		subarray = []
		for j in range(size):
			if i == j:
				subarray.append(1.0)
			else:
				subarray.append(0.0)
		array.append(subarray)

	return matrix(array)

def construct_kernel_matrix(data, gamma):
	array = []
	for i in range(len(data)):
		subarray = []
		for j in range(len(data)):
			subarray.append( gaussian_kernel(data[i], data[j], gamma) )
		array.append(subarray)
	return matrix(array)
def result_hypothesis(beta, gamma, data, x):
	result = 0.0
	for i in range(len(beta)):
		result += beta[i] * gaussian_kernel(data[i], x, gamma)
	return result


y, x = svm_read_feature('data/hw2_lssvm_all.dat')




data_size = 400
Gamma = [32, 2, 0.125]
Lambda = [0.001, 1, 1000]

I = construct_unit_matrix(data_size)

e_list = []

for gam in Gamma:
	e_sublist = []
	for lam in Lambda:

		beta = ((lam * I + construct_kernel_matrix(x[0:data_size], gam)).I * matrix(y[0:data_size]).T).A1
		
		e_in = 0
		for i in range(data_size):
			if result_hypothesis(beta, gam, x[0:data_size], x[i]) * y[i] < 0.0:
				e_in += 1

		e_out = 0
		for i in range(len(y[data_size:])):
			if result_hypothesis(beta, gam, x[0:data_size], x[data_size + i]) * y[data_size + i] < 0.0:
				e_out += 1

		e_sublist.append((float(e_in) / 4.0, float(e_out) * 100.0/ float(len(y[data_size:])) ) )

	e_list.append(e_sublist)



print "============ Results of Q11 & Q12 ============"

for i in range(len(e_list)):
	for j in range(len(e_list[i])):
		e_in, e_out = e_list[i][j]
		print "gamma = %.3f, lambda = %.3f, Ein = %f %%, Eout = %f %%"%(Gamma[i], Lambda[j], e_in, e_out)


print "=============================================="

#test1 = [1, 2, 3]
#test2 = [2, 3, 4]

#print gaussian_kernel(test1, test2, 1)

#print (3 * construct_unit_matrix(3)).I


#testdata = [[1,2], [3,5], [2,4], [9,7] ]
#print construct_kernel_matrix(testdata, 1)


