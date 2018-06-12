## KM (Kuhn-Munkras) Solver

The main function of this solver is to solve the best match of the bipartie graph.

The theory of this algorithm refer to:

* http://blog.sina.com.cn/s/blog_691ce2b701016reh.html 

* http://www.cnblogs.com/wenruo/p/5264235.html 

Hungarian algorithm is the core algorithm in KM. Hungarian algorithm is to find the most number of 
pairs in bipartie graph. But KM is to find the best pairs to maximize the weights of the graph.

Hungarian algorithm can be implemented by DFS or BFS. Two methods were compared in different fully-connected
bipartie graphs. The time spent distribution is shown below,
![dfs vs bfs]()
