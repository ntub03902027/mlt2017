import sys
import math

#import numpy
#from numpy import matrix
#from numpy import linalg
#from numpy import array




class Tree(object):
    def __init__(self):
        self.left = None
        self.right = None
        self.data = None
        self.leafid = -1


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

def is_identical(data):
	identical_x = True
	identical_y = True

	# check identical_x
	x_0 = data[0][0]
	for i in range(len(data)):
		if data[i][0] != x_0:
			identical_x = False
			break

	y_0 = data[0][1]
	for i in range(len(data)):
		if data[i][1] != y_0:
			identical_y = False
			break
	return (identical_y or identical_x)

def get_return_values(data):
	pos_count = 0

	for i in range(len(data)):
		if data[i][1] == +1:
			pos_count += 1
		else:
			pos_count -= 1

	if pos_count >= 0:
		return +1
	return -1


def calculate_gini(data, index, stump):
	coef = [float(index + 1), float(len(data) - index - 1)]
 	
	gini = []

	corr_count = 0
	for subdata in data[:(index + 1)]:
		if subdata[1] == stump[0] * sign(subdata[0][stump[1]] - stump[2]):
			corr_count += 1

	gini.append(1.0 - ( float(corr_count) / float(coef[0]) )**2 - ( float(coef[0] - corr_count) / float(coef[0]) )**2 )

	corr_count = 0
	for subdata in data[(index + 1):]:
		if subdata[1] == stump[0] * sign(subdata[0][stump[1]] - stump[2]):
			corr_count += 1
	gini.append(1.0 - ( float(corr_count) / float(coef[1]) )**2 - ( float(coef[1] - corr_count) / float(coef[1]) )**2 )

	return (coef[0]*gini[0]+coef[1]*gini[1])


#data: zip of (x, y)
#tree.left : true
#tree.right : false
def build_tree(data, idcount = 0):


	# termination condition
	if is_identical(data):
		leaf = Tree()
		leaf.data = get_return_values(data)
		# for pruning
		leaf.leafid = idcount
		idcount += 1

		return (leaf, idcount)


	S = [-1, +1]
	s_opt, i_opt, theta_opt = (0, 0, 0)
	opt_index = 0
	gini_opt = float("inf")

	for i in range(len(data[0][0])):
		data = sorted(data, key = lambda x: x[0][i])

		for j in range(len(data) - 1):
			for s in S:
				theta = 0.5 * (data[j][0][i] + data[j+1][0][i])


				gini = calculate_gini(data, j, (s, i, theta))

				if gini_opt > gini:
					s_opt = s 
					i_opt = i 
					theta_opt = theta 
					opt_index = j
					gini_opt = gini 


	leaf = Tree()
	leaf.data = (s_opt, i_opt, theta_opt)
	data = sorted(data, key = lambda x: x[0][i_opt])
	if s_opt == +1:
		leaf.left, idcount = build_tree(data[opt_index+1:], idcount)
		leaf.right, idcount = build_tree(data[:opt_index+1], idcount)
	else:
		leaf.left, idcount = build_tree(data[:opt_index+1], idcount)
		leaf.right, idcount = build_tree(data[opt_index+1:], idcount)

	return (leaf, idcount)


#(s, i, theta)
def decision(pointx, stump):
	s, i, theta = stump
	if s * sign(pointx[i] - theta) == +1:
		return True
	return False 

def make_decision(pointx, root, prune_leaf, leafid):
	current = root

	while type(current.data) != int:
		if decision(pointx, current.data):
			if prune_leaf and type(current.left.data) == int and leafid == current.left.leafid:
				current = current.right
			else:
				current = current.left
		else:
			if prune_leaf and type(current.right.data) == int and leafid == current.right.leafid:
				current = current.left
			else:
				current = current.right

	return current.data 

def calculate_error(data, root, prune_leaf = False, leafid = -1):
	err_count = 0
	for i in range(len(data)):
		if data[i][1] != make_decision(data[i][0], root, prune_leaf, leafid):
			err_count += 1

	return (float(err_count) / float(len(data)))

def print_tree(root, level):
	if root.left == None and root.right == None:
		if root.data == +1:
			print level*"\t", "return True (%d)"%(root.leafid)
		else:
			print level*"\t", "return False (%d)"%(root.leafid)

	else:
		if root.data[0] == +1:
			print level*"\t", "if x[%d] >= %f:"%(root.data[1], root.data[2])
			print_tree(root.left, level+1)
			print level*"\t", "else:"
			print_tree(root.right, level+1)
		else:
			print level*"\t", "if x[%d] <= %f:"%(root.data[1], root.data[2])
			print_tree(root.left, level+1)
			print level*"\t", "else:"
			print_tree(root.right, level+1)



y, x = read_feature('data/hw3_train.dat')

testy, testx = read_feature('data/hw3_test.dat')



root, leafcount = build_tree(zip(x, y))

print "=============== Results of Q14 ==============="
print_tree(root, 0)
print "=============== Results of Q15 ==============="

print "Ein = %f, Eout = %f"%(calculate_error(zip(x, y), root), calculate_error(zip(testx, testy), root))

print "=============== Results of Q16 ==============="
Ein = []
Eout = []
for i in range(leafcount):
	Ein.append(calculate_error(zip(x, y), root, True, i))
	Eout.append(calculate_error(zip(testx, testy), root, True, i))
	print "Pruning leaf %d, Ein = %f, Eout = %f"%(i, Ein[i], Eout[i])

print "\n==> minimum of Ein = %f, minimum of Eout = %f"%(min(Ein), min(Eout))

print "=============================================="

#print x
#print y

#root = Tree()
#root.data = 123
#root.left = Tree()
#root.left.data = 222
#root.right = Tree()
#root.right.data = 555


#TODO: implement gini index; debug; get stats
