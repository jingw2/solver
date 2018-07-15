## Period Detection Solver

It is to find possible repeated elements by solving minimum Edit Distance with real Penalty (ERP), which is called ERPP (ERP based Period Detection Algorithm).Let's simply describe the algorithm by the following example:

Assume we have a string "ababac", we construct a distance matrix. The values in the main diagonal are all zeros, which will affect the calculation of minimum ERP. Thus, they are changed to infinite. The following graphs show the ERP calculation in period 1 and period 2. 

![p1](https://github.com/jingw2/solver/blob/master/period_detection/p1.png)

For period 1, the element is "a". The origin and destination of ERP is (n-2, n-1) and (0, 1). The route with minimum distance is shown by arrows in the graph. The value is 3. 

![p2](https://github.com/jingw2/solver/blob/master/period_detection/p2.png)

For period 2, the element is "ab". The origin and destination of ERP is (n-3, n-1) and (0, 2). The route with minimum distance is shown by arrows in the graph. The value is 1. 

In general, for period p, we need to find the minimum route from (n-p-1, n-1) to (0, p). The confidence can be calculated in terms of minimum ERP, 

![equation](https://github.com/jingw2/solver/blob/master/period_detection/confidence.gif)


Solver Arguments:
* s (list, tuple or string)
* threshold (confidence threshold)
* method (recursion or dp), for big length of s, please use dp. By default, it is dp.

```Python
import period_detect

s = "ababac"
result = period_detect.solve(s, threshold = 0.7, method = "dp")

## result = {"ab" : 0.75}
```


Reference link: 

* https://wenku.baidu.com/view/8ad300afb8f67c1cfad6b87a.html
