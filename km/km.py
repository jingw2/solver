#!/usr/bin/python 3.6
#-*-coding:utf-8-*-

'''
KM (Kuhn-Munkras) algorithm solver

Reference link:

* http://www.cnblogs.com/wenruo/p/5264235.html               --> algorithm implementation and example 
* http://blog.sina.com.cn/s/blog_691ce2b701016reh.html       --> algorithm theory intro
* https://blog.csdn.net/dark_scope/article/details/8880547   --> Hungarian algorithm intro

Date: 06/10/2018
Author: Jing Wang (jingw2@foxmail.com)

Example Use:
#################################################
import random
import km

## left and right vertice number
numLeft = 4
numRight = 5

## construct graph
graph = np.zeros((numLeft, numRight))
for i in range(numLeft):
	for j in range(numRight):
		graph[i, j] = random.choice(list(range(10)))

## solve
match, weight = km.solve(graph, verbose = 0, method = 'bfs')
#################################################
'''

import numpy as np 
import os
from collections import deque
import random
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=None)
parser.add_argument('-left', type=int,
                    default = 3)
parser.add_argument('-right', type=int,
                    default = 3)
parser.add_argument('-method', type=str,
                    default = 'dfs')
args = parser.parse_args()

## construct graph
random.seed(123)
# prev = [(10, 19, 0), (20, 30, 1), (8, 19, 2), (15, 25, 0)]
# right = [(8, 19, 0, 1), (20, 30, 1, 1), (8, 19, 2, 1), (15, 26, 0, 1), (10, 19, 0, 2)]
# numLeft = args.left
# numRight = args.right
# global graph
# graph = np.zeros((numLeft, numRight))
# # graph[0, 0] = 3 
# # graph[0, 2] = 4
# # graph[1, 0] = 2
# # graph[1, 1] = 1
# # graph[1, 2] = 3
# # graph[2, 2] = 5

# weightList = list(range(1, numLeft * numRight + 1))


def dfs(left, graph):
	'''
	depth first search method

	Args:
	* left (int): the left element index
	* graph (array like): graph to solve

	Return:
	* boolean : if match is found, return True, otherwise False
	'''

	## visited
	visitedLeft[left] = True

	for right in range(numRight):

		if visitedRight[right]: continue # every round, every right can only be retrieved once
		gap = leftExpect[left] + rightExpect[right] - graph[left, right]

		if gap == 0: # match expectation
			visitedRight[right] = True

			if match[right] == -1 or dfs(match[right], graph): # if right has no match or the matched left can find other rights
				match[right] = left
				return True

		else:
			slack[right] = min(slack[right], gap)

	return False

def bfs(left, graph):
	'''
	breath first search method

	Args:
	* left (int): the left element index
	* graph (array like): graph to solve

	Return:
	* boolean : if match is found, return True, otherwise False
	'''
	
	visitedLeft[left] = True

	queue.append(left) # push to the end
	prev[left] = -1
	flag = False # has found expand path

	while (len(queue) != 0 and not flag):
		firstEle = queue[0]
		for right in range(numRight):
			if flag: break
			if visitedRight[right] or graph[firstEle, right] == 0: continue
			gap = leftExpect[firstEle] + rightExpect[right] - graph[firstEle, right]
		
			if gap == 0:

				## push new vertice
				queue.append(match[right])
				visitedRight[right] = True
				
				if match[right] != -1: # find
					visitedLeft[match[right]] = True
					prev[match[right]] = firstEle
				else:
					# recursion 
					flag = True
					d = firstEle
					e = right
					while d != -1:
						t = matchLeft[d]
						matchLeft[d] = e
						match[e] = d
						d = prev[d]
						e = t
		
		queue.popleft() # remove the first element
				
	if matchLeft[left] != -1:
		return True
	else:
		## slack = min{(x, y) | Lx(x) + Ly(y) - W(x, y), x in S, y not in T}, S is visited left, T is not visited right
		for left in range(numLeft):
			if not visitedLeft[left]: continue
			for right in range(numRight):
				if visitedRight[right]: continue
				gap = leftExpect[left] + rightExpect[right] - graph[left, right]
				if gap == 0: continue
				slack[right] = min(slack[right], gap)
		return False
		
def solve(graph, verbose = 0, method = 'dfs'):

	'''
	KM algorithm solver

	Args:
	* graph (np.array like): 
		every row represents the left vertices of bipartie graph
		every column represents the right vertices of bipartie graph
	* verbose (boolean): 1 to show print
	* method: (str): which method to use, dfs or bfs

	Return:
	* match (dict): key is the right element, if value = -1, the right has no match
					value is the matched left element
	* weight (float): total weights of matched graph

	Raise:
	feasibility error
	'''

	## check graph
	global numLeft, numRight
	numLeft, numRight = graph.shape
	if numLeft < numRight:
		raise Exception('Please check the shape graph: {}! Ncols should bigger than nrows.'.format(graph,shape))

	## initialize
	global leftExpect, rightExpect, visitedLeft, visitedRight, match, slack, matchLeft, prev, queue
	leftExpect = {g : np.max(graph[g]) for g in range(numLeft)}
	rightExpect = {b : 0 for b in range(numRight)}
	match = {b: -1 for b in range(numRight)} ## for rights
	matchLeft = {a: -1 for a in range(numLeft)}
	prev = {l : -1 for l in range(numLeft)}
	queue = deque() # for bfs

	# find match for every left 
	for lix in range(numLeft):

		slack = {b : float('inf') for b in range(numRight)} # how many expectation value needs for rights to match 
		while True:
			# if left has no match, lower the expectation value util match is found

			## initialize every round
			visitedLeft = {g : False for g in range(numLeft)}
			visitedRight = {b : False for b in range(numRight)} 

			if method == 'dfs':
				if dfs(lix, graph):
					break # find match
			else:
				if matchLeft[lix] == -1:
					while len(queue) != 0: queue.pop()
					if bfs(lix, graph):
						break # find match
			
			##### cannot find match

			## find the minimum value to decrease
			diff = float('inf')
			for right in range(numRight):
				if not visitedRight[right]:
					diff = min(slack[right], diff)


			## all retrived lefts should decrease expectation value 
			for left in range(numLeft):
				if visitedLeft[left]:
					leftExpect[left] -= diff

			## keep c[x] + c[y] = weight[(x, y)]
			for right in range(numRight):
				# if over one left can match with this right
				if visitedRight[right]:
					rightExpect[right] += diff
				else:
					slack[right] -= diff

		if verbose:
			print('Finish to match left {}'.format(lix))

	## output maximum weights
	weight = 0
	for right, left in match.items():
		if verbose:
			print('left {}, right {}'.format(left, right))
		weight += graph[left, right]

	if verbose:
		print('Maximum match weights: ', weight)

	return match, weight

if __name__ == '__main__':
	from time import time

	timeSpent = {'dfs': [], 'bfs': []}
	sizeList = range(10, 301)
	for m in ['dfs', 'bfs']:
		for size in sizeList:
			numLeft, numRight = size, size
			graph = np.zeros((numLeft, numRight))
			for i in range(numLeft):
				for j in range(numRight):
					graph[i, j] = random.choice(list(range(100)))
			start = time()
			solve(graph, verbose = 0, method = m)
			end = time()
			timeSpent[m].append(abs(end - start))
			print('method {}, size {}'.format(m, size))

	plt.figure()
	# plt.subplot(211)
	plt.plot(sizeList, timeSpent['dfs'], 'r-')
	# plt.title('DFS')

	# plt.subplot(212)
	plt.plot(sizeList, timeSpent['bfs'], 'b-')
	# plt.title('BFS')
	plt.title('KM algorithm in different methods')
	plt.xlabel('graph size')
	plt.ylabel('time spent')
	plt.legend(['DFS', 'BFS'])
	plt.show()





	
	
