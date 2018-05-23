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


y, x = svm_read_feature('data/features.train', 8)  

#testy, testx = svm_read_feature('data/features.test', 8)


C = [0.00001, 0.001, 0.1, 10, 1000]
ein_list = []
nSV_list = []

for c in C:

	m = svm_train(y, x, '-h 0 -s 0 -t 1 -g 1 -r 1 -d 2 -c %f'%(c))

	coef = m.get_sv_coef()
	sv = m.get_SV()
	nSV = len(sv)

	p_label, p_acc, p_val = svm_predict(y, x, m)
	print "Ein = ", 100.0-p_acc[0], "%"

	ein_list.append(100.0-p_acc[0])
	nSV_list.append(nSV)

print "============ Results of Q12 & Q13 ============"
for i in range(len(C)):
	print "log C = %.0f, Ein = %f %%, #SV = %d"%(math.log(C[i], 10), ein_list[i], nSV_list[i])
print "=============================================="

