import sys
import math

# For Bagging
import random
from random import randint


class Tree(object):
    def __init__(self):
        self.left = None
        self.right = None
        self.data = None


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
def build_tree(data):


	# termination condition
	if is_identical(data):
		leaf = Tree()
		leaf.data = get_return_values(data)

		return leaf


	#S = [-1, +1]
	S = [+1]
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
		leaf.left = build_tree(data[opt_index+1:])
		leaf.right = build_tree(data[:opt_index+1])
	else:
		leaf.left = build_tree(data[:opt_index+1])
		leaf.right = build_tree(data[opt_index+1:])

	return leaf

def build_stump(data):

	S = [+1]
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

	err_count = 0

	for point in data[opt_index+1:]:
		if point[1] == -1:
			err_count += 1
	for point in data[:opt_index+1]:
		if point[1] == +1:
			err_count += 1
	if err_count <= len(data) - err_count:
		leaf.left = Tree()
		leaf.left.data = +1
		leaf.right = Tree()
		leaf.right.data = -1
	else:
		leaf.left = Tree()
		leaf.left.data = -1
		leaf.right = Tree()
		leaf.right.data = +1
	return leaf

#(s, i, theta)
def decision(pointx, stump):
	s, i, theta = stump
	if s * sign(pointx[i] - theta) == +1:
		return True
	return False 

def make_decision(pointx, root):
	current = root

	while type(current.data) != int:
		if decision(pointx, current.data):
				current = current.left
		else:
				current = current.right

	return current.data 

def calculate_error(data, root):
	err_count = 0
	for i in range(len(data)):
		if data[i][1] != make_decision(data[i][0], root):
			err_count += 1

	return (float(err_count) / float(len(data)))

def calculate_forest_error(data, rootl):
	err_count = 0
	for i in range(len(data)):
		vote = 0
		for j in range(len(rootl)):
			if make_decision(data[i][0], rootl[j]) == +1:
				vote += 1
			else:
				vote -= 1
		if (vote >= 0 and data[i][1] < 0) or (vote < 0 and data[i][1] >= 0):
			err_count += 1
	return (float(err_count) / float(len(data)))

def update_forest_vote(data, vote, root):
	for i in range(len(data)):
		if make_decision(data[i][0], root) == +1:
			vote[i] += 1
		else:
			vote[i] -= 1
	return vote 

def update_forest_error(data, vote):
	err_count = 0
	for i in range(len(data)):
		if (vote[i] >= 0 and data[i][1] < 0) or (vote[i] < 0 and data[i][1] >= 0):
			err_count += 1
	return  (float(err_count) / float(len(data)))


def print_tree(root, level = 0):
	if root.left == None and root.right == None:
		if root.data == +1:
			print level*"\t", "return True"
		else:
			print level*"\t", "return False"

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

def bootstrap(data, size = -1):
	if size == -1:
		size = len(data)
	bootstrapped = []

	for i in range(size):
		r = randint(0, size - 1)
		bootstrapped.append(data[r])

	return bootstrapped

def print_help():
	print "usage: python %s [problem nos. (only 12-16 are allowed)]"%(sys.argv[0])
	print "example: python %s 12,14-16 (print results of problem 12, 14, 15 and 16) (separated by ',' and no spaces between)"%(sys.argv[0])
	exit()

def parse_arg(arg):
	problist = arg.split(",")

	problems = [False, False, False, False, False]

	for substr in problist:
		if '-' in substr:
			ran = substr.split("-")
			if not len(ran) == 2:
				print_help()
			try:
				int(ran[0])
				int(ran[1])
			except:
				print_help()

			if int(ran[0]) < 12 or int(ran[0]) > 16 or int(ran[1]) < 12 or int(ran[1]) > 16 or int(ran[0]) > int(ran[1]):
				print_help()
			i = int(ran[0])
			while i <= int(ran[1]):
				problems[i-12]=True
				i += 1
		else:
			try:
				int(substr)
			except:
				print_help()

			if int(substr) < 12 or int(substr) > 16:
				print_help()
			problems[int(substr)-12]=True
	return problems

def print_progress(t, T):
	non_space = (t / (T/40)) *"#"
	space = (40 - (t / (T/40))) *" "
	if t < T:
		sys.stdout.write("\rProcessing: [%s%s] (%3d%%)" % (non_space, space, int(t/(T/100))) )
		sys.stdout.flush()
	else:
		sys.stdout.write("\rProcessing: [########################################] (DONE)")
		sys.stdout.flush()
		sys.stdout.write("\n")
		sys.stdout.flush()

# preprocessing
problems = []
inform = []

if len(sys.argv) == 2:
	problems = parse_arg(sys.argv[1])
elif len(sys.argv) == 1:
	problems = [True, True, True, True, True]
else:
	print_help()

for i in range(len(problems)):
	if problems[i] == True:
		inform.append( int(i+12) )
print "Processing results of problems: ", inform





y, x = read_feature('data/hw3_train.dat')

testy, testx = read_feature('data/hw3_test.dat')


# for convenience
train_tuple = zip(x, y)
test_tuple = zip(testx, testy)


train_vote = len(train_tuple) * [0]
test_vote = len(test_tuple) * [0]

train_stump_vote = len(train_tuple) * [0]
test_stump_vote = len(test_tuple) * [0]

T = 30000
bootstrap_size = len(x)

Ein_g = []
Ein_G = []
Eout_G = []

Estump_in_G = []
Estump_out_G = []


print "Number of trees (T) = %d"%(T)
for t in range(T):
	if (t*100) % (T) == 0:
		print_progress(t, T)


	bootstrap_data = bootstrap(train_tuple)
	if problems[0] or problems[1] or problems[2]:
		root = build_tree(bootstrap_data)
	if problems[3] or problems[4]:
		stump = build_stump(bootstrap_data)

	# Q12
	if problems[0]:
		Ein_g.append(calculate_error(train_tuple, root))

	# Q13 
	if problems[1]:
		train_vote = update_forest_vote(train_tuple, train_vote, root)
		Ein_G.append(update_forest_error(train_tuple, train_vote))

	# Q14
	if problems[2]:
		test_vote = update_forest_vote(test_tuple, test_vote, root)
		Eout_G.append(update_forest_error(test_tuple, test_vote))

	# Q15
	if problems[3]:
		train_stump_vote = update_forest_vote(train_tuple, train_stump_vote, stump)
		Estump_in_G.append(update_forest_error(train_tuple, train_stump_vote))

	# Q16
	if problems[4]:
		test_stump_vote = update_forest_vote(test_tuple, test_stump_vote, stump)
		Estump_out_G.append(update_forest_error(test_tuple, test_stump_vote))

print_progress(T, T)

if problems[0]:
	print "=============== Results of Q12 ==============="
	print "Ein_g = ", Ein_g

if problems[1]:
	print "=============== Results of Q13 ==============="
	print "(Random Forest) Ein_G = ", Ein_G

if problems[2]:
	print "=============== Results of Q14 ==============="
	print "(Random Forest) Eout_G = ", Eout_G

if problems[3]:
	print "=============== Results of Q15 ==============="
	print "(Random Decision Stump) Ein_G = ", Estump_in_G

if problems[4]:
	print "=============== Results of Q16 ==============="
	print "(Random Decision Stump) Eout_G = ", Estump_out_G


print "=============================================="



#print x
#print y


