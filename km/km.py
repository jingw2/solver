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

## Note that, match is a dictionary, with key the index of left
value is the index of matched right
#################################################
'''

import numpy as np 
import os
from collections import deque
import random
import argparse
import matplotlib.pyplot as plt

def dfs(left, graph, is_constraint_on_weight):
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
		if is_constraint_on_weight:
			if graph[left, right] == 0: continue
		if visitedRight[right]: continue # every round, every right can only be retrieved once
		gap = leftExpect[left] + rightExpect[right] - graph[left, right]

		if gap == 0: # match expectation
			visitedRight[right] = True

			# if right has no match or the matched left can find other rights
			if match[right] == -1 or dfs(match[right], graph, is_constraint_on_weight): 
				match[right] = left
				return True

		else: # to accelerate
			slack[right] = min(slack[right], gap)

	return False

def bfs(left, graph, is_constraint_on_weight):
	'''
	breath first search method

	Args:
	* left (int): the left element index
	* graph (array like): graph to solve
	* is_constraint_on_weight (boolean)

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
			if is_constraint_on_weight:
				if graph[firstEle, right] == 0: continue
			if visitedRight[right]: continue
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
		
def solve(graph, verbose = 0, method = 'dfs', is_constraint_on_weight=True):

	'''
	KM algorithm solver

	Args:
	* graph (np.array like): 
		every row represents the left vertices of bipartie graph
		every column represents the right vertices of bipartie graph
	* verbose (boolean): 1 to show print
	* method: (str): which method to use, dfs or bfs
	* is_constraint_on_weight (boolean): 
		want to constrain on weight, impossible match on weight = 0 edge

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
	is_transpose = False
	if numLeft > numRight:
		print("Left is greater than right, transpose graph matrix")
		graph = graph.T
		numLeft, numRight = graph.shape
		is_transpose = True

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
				if dfs(lix, graph, is_constraint_on_weight):
					break # find match
			else:
				if matchLeft[lix] == -1:
					while len(queue) != 0: queue.pop()
					if bfs(lix, graph, is_constraint_on_weight):
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
	out = {}
	for right, left in match.items():
		if verbose:
			print('left {}, right {}'.format(left, right))
		weight += graph[left, right]
		if left != -1:
			if is_transpose: # exchange the order
				out[right] = left
			else:
				out[left] = right

	if verbose:
		print('Maximum match weights: ', weight)

	return out, weight





	
	
