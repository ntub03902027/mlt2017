import sys
sys.path.append('./libsvm/')
from svmutil import *
import math
import random


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


Gamma = [0.1, 1, 10, 100, 1000]

poll_list = [0, 0, 0, 0, 0]

total_interations = 100

for i in range(total_interations):
# shuffle, decide train and validation
	train_x = x
	train_y = y
	train_zip = zip(train_x, train_y)
	random.shuffle(train_zip)
	train_x, train_y = zip(*train_zip)

	val_x = train_x[:1000]
	val_y = train_y[:1000]
	train_x = train_x[1000:]
	train_y = train_y[1000:]

	eval_list = []

	for gamma in Gamma:
		m = svm_train(train_y, train_x, '-h 0 -s 0 -t 2 -g %f -c 0.1'%(gamma))

		p_label, p_acc, p_val = svm_predict(val_y, val_x, m)
		e_val = 100.0 - p_acc[0]

		print "Iteration #%.0f: log gamma = %.0f, Eval = %f %%"%(i, math.log(gamma, 10), e_val)
		eval_list.append(e_val)

	winner = eval_list.index(min(eval_list))

	print "Result of #%d: choose log gamma = %.0f"%(i, math.log(Gamma[winner], 10))
	poll_list[winner] += 1


print "=============== Results of Q16 ==============="
for i in range(len(Gamma)):
	print "log gamma = %.0f, seleted = %d (%.2f %%)"%(math.log(Gamma[i], 10), poll_list[i], float(poll_list[i]) / float(total_interations) * 100.0)
print "=============================================="