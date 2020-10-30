# /usr/bin/env python 3.6
# -*-coding:utf-8-*-

'''
Period Detection Solver

Reference link:
https://wenku.baidu.com/view/8ad300afb8f67c1cfad6b87a.html

Author: Jing Wang (jingw2@foxmail.com)
'''
import numpy as np 

## algorithm
def recurse(n, m):
	'''
	recursion method
	find minimum ERP distance from (n - p - 1, n - 1) to (0, p)

	Args:
	n (int): starting row index
	m (int): starting column index

	Return:
	d (int): minimum ERP distance
	'''
	cache = {}
	d = 0
	if (n, m) in cache:
		return cache[(n, m)]
	if n == 0 and m == p:
		d += matrix[n][m]
	elif n == 0 and m > p:
		d += recurse(n, m - 1) + matrix[n, m]
	else:
		d += min([recurse(n-1, m-1), recurse(n-1, m)]) + matrix[n][m]
	cache[(n, m)] = d
	return d

def dp(n, m):
	'''
	dynamic programming
	find minimum ERP distance from (n - p - 1, n - 1) to (0, p)

	Args:
	n (int): starting row index
	m (int): starting column index

	Return:
	minimum ERP distance
	'''
	nr, nc = matrix.shape
	d = np.zeros((nr, nc))
	for i in range(n, -1, -1):
		for j in range(m, p - 1, -1):
			if i < nr - 1 and j < nc - 1:
				valid = []
				if (j - i - 1) >= (m - n):
					valid.append(d[i + 1, j])
				if (j + 1 - i) >= (m - n):
					valid.append(d[i, j + 1])
				if (j - i) >= (m - n):
					valid.append(d[i + 1, j + 1])
				if len(valid) > 0:
					d[i, j] = min(valid) + matrix[i][j]
				else:
					d[i, j] = matrix[i][j]
			elif i < nr - 1 and j == nc - 1:
				if (j - i - 1) >= (m - n):
					d[i, j] = d[i + 1, j]+ matrix[i][j]
				else:
					d[i, j] = matrix[i, j]
			elif i == nr - 1 and j < nc - 1:
				if (j + 1 - i) >= (m - n):
					d[i, j] = d[i, j + 1] + matrix[i][j]
				else:
					d[i, j] = matrix[i, j]


	return d[0, p]

def solve(s, threshold, method = "dp"):
	'''
	solve function
	'''

	# check
	if len(s) == 0 or len(s) == 1:
		return None

	try:
		s[0]
		s[0:]
	except:
		raise Exception("Please make sure input can be sliced!")

	# generate distance matrix
	global matrix, p
	n = len(s)
	matrix = np.zeros((n, n))
	for i in range(n):
		for j in range(n):
			if i == j:
				matrix[i, j] = float("inf") # leave the main diagonal
				continue
			if s[i] == s[j]:
				matrix[i, j] = 0
			else:
				matrix[i, j] = 1

	result = {}
	for p in range(1, n // 2 + 1):
		if method == "dp":
			d = dp(n - p - 1, n - 1)
		else:
			d = recurse(len(s) - p - 1, n - 1)
		confidence = (n - p - d) / (n - p)

		if confidence > threshold:
			result[tuple(s[:p])] = round(confidence, 3)
	
	return result


s = "ababac"
if __name__ == '__main__':
	print(solve(s, 0.7))
