import sys
sys.path.append('./libsvm/')
from svmutil import *
import math


def svm_read_feature(data_file_name, digit):
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
		line = line.split(None, 1)
		#print line
		# In case an instance with all zero features
		if len(line) == 1: line += ['']
		label, features = line
		#parse prob_x
		xi = {}
		ind = 1
		for e in features.split():
			xi[ind] = float(e)
			ind += 1
		#parse prob_y
		if int(float(label)) == digit:
			prob_y += [float(+1)]
		else:
			prob_y += [float(-1)]
		prob_x += [xi]
	return (prob_y, prob_x)


y, x = svm_read_feature('data/features.train', 0)  

testy, testx = svm_read_feature('data/features.test', 0)

Gamma = [1, 10, 100, 1000, 10000]

eout_list = []

for gamma in Gamma:
	m = svm_train(y, x, '-h 0 -s 0 -t 2 -g %d -c 0.1'%(gamma))

#	coef = m.get_sv_coef()
#	sv = m.get_SV()
#	nSV = len(sv)

	p_label, p_acc, p_val = svm_predict(testy, testx, m)
	print "Eout = ", 100.0-p_acc[0], "%"
	eout_list.append(100.0-p_acc[0])
	

print "=============== Results of Q15 ==============="
for i in range(len(Gamma)):
	print "log 10 gamma = %.0f, Eout = %f %%"%(math.log(Gamma[i], 10), eout_list[i])
print "=============================================="

