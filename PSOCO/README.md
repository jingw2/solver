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
![equation]("https://latex.codecogs.com/svg.latex?\begin{align*}&space;\min&space;\&space;f(x)&space;&=&space;(x_1&space;-&space;2)^2&space;&plus;&space;(x_2&space;-&space;1)^2&space;\\&space;s.t.&space;\&space;x_1&space;&=&space;x_2&space;-&space;1&space;\\&space;x_1^2/4&&plus;x_2^2-1&space;\leq&space;0&space;\end{align*}")

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
