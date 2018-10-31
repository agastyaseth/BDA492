
# UV Decomposition Example
## Assignment 2: Question 1

### Introduction

The simplest form of matrix decomposition is to find a pair of matrixes, the first (p) with few columns and the second (q) with few rows, whose product is close to the given matrix M.

### Intuition

The axes of the subspace can be chosen by:
* The first dimension is the direction in which the points exhibit the greatest variance.
* The second dimension is the direction, orthogonal to the first, in which points show the greatest variance.
* And so on…, until you have “enough” dimensions.

### Measuring the Error (RSME)
* Common way to evaluate how well P = UV approximates M is by RMSE (root-mean-square error).
* Average $(m_{ij} – p_{ij})^2$ over all i and j.
* Take the square root.
* Square-rooting changes the scale of error, but doesn’t really effect which choice of U and V is best

### Gradient Descent

* Pick r, the number of latent factors.
* Think of U and V as composed of variables, uik and vkj.
* Express the RMSE as (the square root of): $$e =  \sum_{ij} (r_{ij} – \sum_k p_{ik} q_{kj})^2 $$
* Gradient descent: repeatedly find the derivative of E with respect to each variable and move each a small amount in the direction that lowers the value of E.


### Regularization

The above algorithm is a very basic algorithm for factorizing a matrix. There are a lot of methods to make things look more complicated. A common extension to this basic algorithm is to introduce regularization to avoid overfitting. This is done by adding a parameter $\beta$ and modify the squared error as follows:

$$e_{ij}^2 = (r_{ij}- \sum_{k=1}^K p_{ik}q_{kj})^2 + \beta/2 \sum_{k=1}^K (||P||^2 + ||Q||^2) $$

### How to run

Running the program is really straightforward:

* Open the .ipynb file in Jupyter Notebook
* Run all the cells

### Importing Libraries


```python
import numpy as np
```

### UV Decomposition Function


```python
def uvd(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02): # alpha = learning rate; beta = regularizing parameter
    Q = Q.T
    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i,j] > 0:
                    eij = R[i,j] - np.dot(P[i,:],Q[:,j])
                    for k in range(K):
                        P[i,k] = P[i,k] + alpha * (2 * eij * Q[k,j] - beta * P[i,k])
                        Q[k,j] = Q[k,j] + alpha * (2 * eij * P[i,k] - beta * Q[k,j])
        eR = np.dot(P,Q)
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i,j] > 0:
                    e = e + pow(R[i,j] - np.dot(P[i,:],Q[:,j]), 2)
                    for k in range(K):
                        e = e + (beta/2) * (pow(P[i,k],2) + pow(Q[k,j],2))
        if e < 0.001:
            break
    return P, Q.T
```

### Working out an Example


```python
R = [[5,3,0,1],[4,0,0,1],[1,1,0,5],[1,0,0,4],[0,1,5,4],] #A sparse matrix with missing values as '0'
R = np.array(R)
R
```




    array([[5, 3, 0, 1],
           [4, 0, 0, 1],
           [1, 1, 0, 5],
           [1, 0, 0, 4],
           [0, 1, 5, 4]])




```python
N = len(R)
M = len(R[0])
K = 2

#assigning random matrices for P and Q
P = numpy.random.rand(N,K) 
Q = numpy.random.rand(M,K)

nP, nQ = uvd(R, P, Q, K)
nR = np.dot(nP, nQ.T)
nR
```




    array([[4.97727843, 2.98467942, 0.8529939 , 1.00006764],
           [3.98192378, 2.40638611, 0.92702492, 0.99732324],
           [0.98727648, 1.03903922, 6.05359292, 4.94338625],
           [1.00793613, 0.95913662, 4.84221357, 3.96785905],
           [0.9862434 , 0.95471   , 4.95146456, 4.05459523]])



**We see that we have a matrix that closely resembles the original matrix R, with a good approximation of the missing values.**
