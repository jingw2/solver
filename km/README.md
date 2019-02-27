## KM (Kuhn-Munkras) Solver

The main function of this solver is to solve the best match of the bipartie graph.

The theory of this algorithm refer to:

* http://blog.sina.com.cn/s/blog_691ce2b701016reh.html 

* http://www.cnblogs.com/wenruo/p/5264235.html 

Hungarian algorithm is the core algorithm in KM. Hungarian algorithm is to find the most number of 
pairs in bipartie graph. But KM is to find the best pairs to maximize the weights of the graph.

Hungarian algorithm can be implemented by DFS or BFS. Two methods were compared in different fully-connected
bipartie graphs. The time spent distribution is shown below,
![KM](https://raw.githubusercontent.com/jingw2/solver/master/km/dfs%20vs%20bfs.png)

It can be seen that dfs method is little better than bfs when the size is small, but bfs is obviously faster than
dfs with the size growing.

Usage:
```python
import numpy as np
import km

# create a graph 
graph = np.random.randn(3, 3)

# solve using km solver
match, totWeight = km.solve(graph, method = "bfs", verbose = 0, is_constraint_on_weight=True)

# match is the dictionary, key is the right index, value is 
# the matched left index, or -1, which is no match. 

# Argument:
# * graph (np.array like): 
#   every row represents the left vertices of bipartie graph
#   every column represents the right vertices of bipartie graph
# * verbose (boolean): 1 to show print
# * method: (str): which method to use, dfs or bfs
# * is_constraint_on_weight (boolean): 
#   want to constrain on weight, impossible match on weight = 0 edge
```
