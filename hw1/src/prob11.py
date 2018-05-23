import sys
sys.path.append('./libsvm/')
from svmutil import *
import math


#x = [{1: 1.0, 2: 0.0}, {1: 0.0, 2: 1.0}, {1: 0.0, 2: -1.0}, {1: -1.0, 2: 0.0}, {1: 0.0, 2: 2.0}, {1: 0.0, 2: -2.0}, {1: -2.0, 2: 0.0}]
#y = [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0]

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


#a, b = svm_read_problem('data/a1a')

#m = svm_train(y, x, '-s 1 -t 1 -g 1 -r 2 -d 2 -c 1000000 -h 0')

C = [0.00001, 0.001, 0.1, 10, 1000]
w_list = []

for c in C:

	m = svm_train(y, x, '-h 0 -s 0 -t 0 -c %f'%(c))

	coef = m.get_sv_coef()
	sv = m.get_SV()
	w = [0.0, 0.0]
	nSV = len(sv)
	for i in range(nSV):
		w[0] += coef[i][0] * sv[i][1]
		w[1] += coef[i][0] * sv[i][2]

	w_norm = math.sqrt(w[0]**2+w[1]**2)
	print "w =", w
	print "||w|| =", w_norm

	w_list.append(w_norm)

print "=============== Results of Q11 ==============="
for i in range(len(C)):
	print "log C = %.0f, ||w|| = %f"%(math.log(C[i], 10), w_list[i])
print "=============================================="

#print m.get_sv_coef()
#print m.get_SV()

