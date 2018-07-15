# /usr/bin/env python 3.6
# -*-coding:utf-8-*-

'''
Conjugate Gradient Method

Reference link:
https://en.wikipedia.org/wiki/Conjugate_gradient_method

Author: Jing Wang
'''

import numpy as np 
from copy import deepcopy
import random

random.seed(123)

def solve(A, b, max_iter):
	'''
	Args:

	A (array): should be positive definite
	b (array): 
	'''
	
	if A.shape[0] != b.shape[0]:
		raise Exception("Please check the shape of array!")

	threshold = 1e-10
	r = deepcopy(b)
	p = deepcopy(b)
	k = 0
	x = np.zeros_like(b)
	while k < max_iter:

		rdot = r.T.dot(r)
		Ap = A.dot(p)

		alpha = rdot / (p.T.dot(Ap))
		x = x + alpha * p
		r = r - alpha * Ap

		newrdot = r.T.dot(r)
		if np.sqrt(newrdot) < threshold:
			break

		beta = newrdot / rdot

		p = r + beta * p

		k += 1
	return x

if __name__ == '__main__':

	A = np.array([[4, 1], [1, 3]])
	b = np.array([[1], [2]])

	print("A: ", A)
	print("b: ", b)

	x = np.linalg.inv(A).dot(b)
	x2 = solve(A, b, 10)

	print("x: ", x)
	print("x2: ", x2)