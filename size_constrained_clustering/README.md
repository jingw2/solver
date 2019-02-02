## Size Constrained Clustering Solver

Implementation of [Deterministic Annealing](http://web.eecs.umich.edu/~mayankb/docs/ClusterCap.pdf)
for size constrained clustering. Constrained optimization in size constrained clustering is also included. 

Size constrained clustering can be treated as an optimization problem, such that 

![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cbegin%7Balign*%7D%20%5Cmin_%7By_i%2C%20j%20%5Cin%20%5B1%2C%20%7C%5Cmathcal%7BC%7D%7C%5D%7D%20%5Csum_%7Bi%3D1%7D%5E%7B%7C%5Cmathcal%7BR%7D%7C%7D%20p%28x_i%29%20%26%5Cleft%5C%7B%5Cmin_%7Bj%20%5Cin%20%5B1%2C%20%7C%5Cmathcal%7BC%7D%7C%5D%7D%20d%28x_i%2C%20y_j%29%20%5Cright%5C%7D%20%5C%5C%20s.t.%20%5Chspace%7B1cm%7D%20%5Csum_%7Bi%3D1%7D%5E%7B%7C%5Cmathcal%7BR%7D%7C%7D%20p%28x_i%29%5Ccdot%20z_%7Bi%2Cj%7D%20%26%5Cleq%20%5Clambda_j%20&plus;%20%5Cepsilon%2C%20%5Chspace%7B0.5cm%7D%20j%20%5Cin%20%5B1%2C%20%7C%5Cmathcal%7BC%7D%7C%5D%20%5C%5C%20%5Ctext%7Bwhere%7D%20%5Chspace%7B0.5cm%7D%20z_%7Bi%2Cj%7D%20%26%3D%20%5Cbegin%7Bcases%7D%201%20%26%5Chspace%7B0.2cm%7D%20x_i%20%5Cin%20%5Ctext%7BCluster%7D_j%20%5C%5C%200%20%26%5Chspace%7B0.2cm%7D%20%5Ctext%7Botherwise%7D%20%5Cend%7Bcases%7D%20%5C%5C%20%5Csum_%7Bj%7D%5E%7B%7C%5Cmathcal%7BC%7D%7C%7D%20%26%20%5Clambda_j%20%3D%201%20%5Chspace%7B0.2cm%7D%20%5Ctext%7Band%7D%20%5Chspace%7B0.2cm%7D%20%5Csum_%7Bj%7D%5E%7B%7C%5Cmathcal%7BC%7D%7C%7D%5Cmathcal%7BC%7D%5Ec_j%20%5Cgeq%20%7C%5Cmathcal%7BR%7D%7C%20%5Cend%7Balign*%7D)

Details in deterministic annealing can be referred in paper. 

### Reference
* [Clustering with Capacity and Size Constraints: A Deterministic
Approach](http://web.eecs.umich.edu/~mayankb/docs/ClusterCap.pdf)
* [Deterministic Annealing, Clustering and Optimization](https://thesis.library.caltech.edu/2858/1/Rose_k_1991.pdf)
