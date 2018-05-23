import sys
import math
import random
from random import randint

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

def kernel(x_1, x_2):
	return numpy.inner(array(x_1), array(x_2)) 

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

def construct_kernel_matrix(data):
	array = []
	for i in range(len(data)):
		subarray = []
		for j in range(len(data)):
			subarray.append( kernel(data[i], data[j]) )
		array.append(subarray)
	return matrix(array)
def result_hypothesis(beta, data, x):
	result = 0.0
	for i in range(len(beta)):
		result += beta[i] * kernel(data[i], x)
	return result

def predict(sample_hypo, x):
	poll = 0
	arrayx = array(x)
	for i in range(len(sample_hypo) ):
		poll += sign(numpy.inner(sample_hypo[i], arrayx))
	return sign(poll)

def sign(x):
	if x > 0:
		return 1
	elif x < 0:
		return -1
	return 0


y, x = svm_read_feature('data/hw2_lssvm_all.dat')




data_size = 400
bag_iter = 200
Lambda = [0.01, 0.1, 1, 10, 100]

I = construct_unit_matrix(data_size)

e_list = []



for lam in Lambda:
	print "Now: lambda = %f"%(lam)
	sample_hypo = []
	for t in range(bag_iter):

		x_sample = []
		y_sample = []

		for i in range(400):
			r = randint(0, 399)
			x_sample.append(x[r])
			y_sample.append(y[r])

		beta = ((lam * I + construct_kernel_matrix(x_sample)).I * matrix(y_sample).T).A1

		sample_hypo.append( array((matrix(beta) * matrix(x_sample)).A1) )

	
	e_in = 0
	for i in range(data_size):
		pred = predict(sample_hypo, x[i])
		if pred != 0 and pred != y[i]:
			e_in += 1

	e_out = 0
	for i in range(len(y[data_size:])):
		pred = predict(sample_hypo, x[data_size + i])
		if pred != 0 and pred != y[data_size + i]:
			e_out += 1

	e_list.append((float(e_in) * 100.0 / float(data_size), float(e_out) * 100.0/ float(len(y[data_size:])) ) )

	#break




print "============ Results of Q15 & Q16 ============"

for i in range(len(e_list)):
	#print e_list[i]
	e_in, e_out = e_list[i]
	print "lambda = %.3f, Ein = %f %%, Eout = %f %%"%(Lambda[i], e_in, e_out)


print "=============================================="

#test1 = [1, 2, 3]
#test2 = [2, 3, 4]

#print gaussian_kernel(test1, test2, 1)

#print (3 * construct_unit_matrix(3)).I


#testdata = [[1,2], [3,5], [2,4], [9,7] ]
#print construct_kernel_matrix(testdata, 1)


