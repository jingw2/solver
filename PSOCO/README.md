## Particle Swarm Optimization Constraint Optimization Solver

### Arguments
|Name |Type|Default Value|
|-----|----|-------------|
|particle_size|int|2000|
|max_iter|int|1000|
|sol_size|int|7|
|fitness|function|null|
|constraints|a list of functions|null|

### Usage
![equation](https://latex.codecogs.com/svg.latex?%5Cbegin%7Balign*%7D%20%5Cmin%20%5C%20f%28x%29%20%26%3D%20%28x_1%20-%202%29%5E2%20&plus;%20%28x_2%20-%201%29%5E2%20%5C%5C%20s.t.%20%5C%20x_1%20%26%3D%20x_2%20-%201%20%5C%5C%20x_1%5E2/4%26&plus;x_2%5E2-1%20%5Cleq%200%20%5Cend%7Balign*%7D)

```python
def objective(x):
    '''create objectives based on inputs x as 2D array'''
    return (x[:, 0] - 2) ** 2 + (x[:, 1] - 1) ** 2 

def constraints1(x):
    '''create constraint1 based on inputs x as 2D array'''
    return x[:, 0] - 2 * x[:, 1] + 1 

def constraints2(x):
    '''create constraint2 based on inputs x as 2D array'''
    return - (x[:, 0] - 2 * x[:, 1] + 1)

def constraints3(x):
    '''create constraint3 based on inputs x as 2D array'''
    return x[:, 0] ** 2 / 4. + x[:, 1] ** 2 - 1
    
constraints = [constraints1, constraints2, constraints3]
num_runs = 10
for _ in range(num_runs):
    psoco = PSOCO(sol_size=2, fitness=objective, constraints=constraints)
    psoco.init_Population()
    psoco.solve()
    # best solutions
    x = psoco.gbest.reshape((1, -1))
```
### Reference
* [Particle Swarm Optimization Method for
Constrained Optimization Problems](https://www.cs.cinvestav.mx/~constraint/papers/eisci.pdf)
