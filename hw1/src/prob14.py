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

#testy, testx = svm_read_feature('data/features.test', 0)

C = [0.001, 0.01, 0.1, 1, 10]

dist_list = []

for c in C:
	m = svm_train(y, x, '-h 0 -s 0 -t 2 -g 80 -c %f'%(c))

	coef = m.get_sv_coef()
	sv = m.get_SV()
	nSV = len(sv)


	# find the first free support vector
	i = 0
	for i in range(len(sv)):
		if abs(coef[i][0]) != c:
			break

	dist = 0.0
	# w^T *x
#	for j in range(len(sv)):
#		dist += coef[j][0] * math.exp(-80 * ((sv[j][1] - sv[i][1])**2 + (sv[j][2] - sv[i][2])**2 ))
#	b = 0.0
#	if coef[i][0] > 0.0:
#		b = 1.0 * m.rho[0]
#	else:
#		b = -1.0 * m.rho[0]
#	dist = abs(dist + b)
	# compute w-normalization
	w_norm = 0.0
	for i in range(len(sv)):
		for j in range(len(sv)):
			w_norm += coef[i][0] * coef[j][0] * math.exp(-80 * ((sv[j][1] - sv[i][1])**2 + (sv[j][2] - sv[i][2])**2 ))
	w_norm = math.sqrt(w_norm)
#	dist = dist / w_norm
	dist = 1.0 / w_norm
#	print dist

	dist_list.append(dist)
	#p_label_in, p_acc_in, p_val_in = svm_predict(y, x, m)

#	p_label, p_acc, p_val = svm_predict(testy, testx, m)
#	print "Eout = ", 100.0-p_acc[0], "%"
	

print "=============== Results of Q14 ==============="
for i in range(len(C)):
	print "log C = %.0f, dist = %f"%(math.log(C[i], 10), dist_list[i])
print "=============================================="
 
