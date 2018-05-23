import numpy
from cvxopt import matrix
from cvxopt import solvers



def transformation(x):
	return matrix([2*(x[1]**2)-4*x[0]+1, x[0]**2-2*x[1]-3])

def kernel(x_1, x_2):
	return (2+x_1[0]*x_2[0]+x_1[1]*x_2[1])**2



y = [-1.0, -1.0, -1.0, +1.0, +1.0, +1.0, +1.0]
x_data = [[1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [-1.0, 0.0], [0.0, 2.0], [0.0, -2.0], [-2.0, 0.0]]

w, h = 7, 7
p_array = [[0 for i in range(w)] for j in range(h)]

for i in range(w):
	for j in range(h):
		p_array[i][j] = y[i]*y[j]*kernel(x_data[i], x_data[j])


P = matrix(numpy.asarray(p_array), tc='d')
q = matrix([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
G = matrix([ [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0] ])
h = matrix([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
A = matrix.trans(matrix(numpy.asarray(y), tc='d'))
b = matrix([0.0])


sol = solvers.qp(P, q, G, h, A, b) 

#print P, q, G, h, A, b


print "Solution (alpha):\n" , sol['x']


alpha_opt = [i for i in sol['x']]
print alpha_opt


alpha_svm = []

for i in alpha_opt:
	if i < 10e-07:
		alpha_svm.append(0.0)
	else:
		alpha_svm.append(i)


b_opt = -1.0
for i in range(7):
	b_opt -= alpha_svm[i] * y[i] * kernel(x_data[i], x_data[1])
print b_opt
