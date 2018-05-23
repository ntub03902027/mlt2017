import numpy
from cvxopt import matrix
from cvxopt import solvers



def transformation(x):
	return matrix([2*(x[1]**2)-4*x[0]+1, x[0]**2-2*x[1]-3])

def phi1(x):
	return 2*(x[1]**2)-4*x[0]+1
#	return x[1]**2-2*x[0]+3
def phi2(x):
	return x[0]**2-2*x[1]-3





rawdata = matrix([[1, 0, -1], [0, 1, -1], [0, -1, -1], [-1, 0, +1], [0, 2, +1], [0, -2, +1], [-2, 0, +1]])

y = matrix([-1, -1, -1, +1, +1, +1, +1])
x_data = [[1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [-1.0, 0.0], [0.0, 2.0], [0.0, -2.0], [-2.0, 0.0]]




P = matrix([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
q = matrix([0.0, 0.0, 0.0])
G = matrix([[-y[0]*1.0, -y[1]*1.0, -y[2]*1.0, -y[3]*1.0, -y[4]*1.0, -y[5]*1.0, -y[6]*1.0], [-y[0]*phi1(x_data[0]), -y[1]*phi1(x_data[1]), -y[2]*phi1(x_data[2]), -y[3]*phi1(x_data[3]), -y[4]*phi1(x_data[4]), -y[5]*phi1(x_data[5]), -y[6]*phi1(x_data[6])], [-y[0]*phi2(x_data[0]), -y[1]*phi2(x_data[1]), -y[2]*phi2(x_data[2]), -y[3]*phi2(x_data[3]), -y[4]*phi2(x_data[4]), -y[5]*phi2(x_data[5]), -y[6]*phi2(x_data[6])]])
h = matrix([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])

sol = solvers.qp(P, q, G, h) 


print G

print sol['x']
