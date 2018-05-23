import sys
sys.path.append('./libsvm/')
from svmutil import *
import math



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
		xi = {}
		ind = 1
		for e in features.split():
			xi[ind] = float(e)
			ind += 1
		#parse prob_y
		prob_y += [float(label)]
		prob_x += [xi]	
	return (prob_y, prob_x)


y, x = svm_read_feature('data/hw2_lssvm_all.dat')

#print y

Gamma = [32, 2, 0.125]
C = [0.001, 1, 1000]

e_list = []

for gamma in Gamma:
	e_sublist = []
	for c in C:

		m = svm_train(y[0:400], x[0:400], '-s 3 -t 2 -h 0 -p 0.5 -g %f -c %f'%(gamma, c))


		p_label, p_acc, p_val = svm_predict(y[0:400], x[0:400], m)
		p_label_out, p_acc_out, p_val_out = svm_predict(y[400:], x[400:], m)

		#print p_label 

		e_in = 0
		for i in range(len(p_label) ):
			if p_label[i] * y[i] < 0.0:
				e_in += 1

		e_out = 0
		for i in range(len(p_label_out)):
			if p_label_out[i] * y[i + 400] < 0.0:
				e_out += 1


		e_sublist.append((float(e_in) / 4.0, float(e_out) * 100.0/ float(len(p_label_out)) ) )

	e_list.append(e_sublist)



print "============ Results of Q13 & Q14 ============"

for i in range(len(e_list)):
	for j in range(len(e_list[i])):
		#print e_list[i][j]
		e_in, e_out = e_list[i][j]
		print "gamma = %.3f, C = %.3f, Ein = %f %%, Eout = %f %%"%(Gamma[i], C[j], e_in, e_out)



print "=============================================="
