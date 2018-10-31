
# Singular Value Decomposition
## Assignment 2: Question 2

### Why Singular Value Decomposition

* Gives a decomposition of any matrix into a product of three matrices.
* There are strong constraints on the form of each of these matrices.
* Results in a decomposition that is essentially unique.
* From this decomposition, you can choose any number r of intermediate concepts (latent factors) in a way that minimizes the RMSE error given that value of r.

### Orthonormal Bases

* Vectors are orthogonal if their dot product is 0.
* Example: [1,2,3].[1,-2,1] = 0, so these two vectors are orthogonal.
* A unit vector is one whose length is 1.
* Length = square root of sum of squares of components.
* No need to take square root if we are looking for length = 1.

**Example: [0.8, -0.1, 0.5, -0.3, 0.1] is a unit vector, since 0.64 + 0.01 + 0.25 + 0.09 + 0.01 = 1.**

An orthonormal basis is a set of unit vectors any two of which are orthogonal.

In this assignment, I'll be discussing a reference example from the following book (pg. 420)
[Mining Massive Datasets](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=2&cad=rja&uact=8&ved=2ahUKEwjfso-Tm7HeAhVDXn0KHdPHC6sQFjABegQIARAC&url=http%3A%2F%2Finfolab.stanford.edu%2F~ullman%2Fmmds%2Fbook.pdf&usg=AOvVaw3zRMVWxsYJ--Uywgu3MQNy)

### How to run

Running this program is really straightforward:

* Open the .ipynb file
* Run all the cells of the notebook

### Initializing the libraries


```python
import numpy as np
import sympy 
from sympy.solvers import solve
```

### Calculating Eigenvalues and Eigenvectors


```python
def eigen(A):
    x = sympy.Symbol('x')
    m = sympy.Matrix(A-x*np.identity(A.shape[1]))
    e = solve(m.det())
    
    
    
    return np.linalg.eig(A)
    
```

### Generating a random matrix


```python
A = np.random.randint(10,size = (5,5))
A
```




    array([[1, 0, 4, 5, 6],
           [0, 1, 9, 2, 2],
           [6, 9, 0, 6, 5],
           [5, 0, 1, 1, 1],
           [0, 4, 5, 7, 6]])




```python
np.linalg.matrix_rank(A)
```




    5



### Calculating U and Sigma


```python
A_t = np.transpose(A)
A_t
```




    array([[1, 0, 6, 5, 0],
           [0, 1, 9, 0, 4],
           [4, 9, 0, 1, 5],
           [5, 2, 6, 1, 7],
           [6, 2, 5, 1, 6]])




```python
AA_t = A.dot(A_t)
AA_t
```




    array([[ 78,  58,  66,  20,  91],
           [ 58,  90,  31,  13,  75],
           [ 66,  31, 178,  41, 108],
           [ 20,  13,  41,  28,  18],
           [ 91,  75, 108,  18, 126]])




```python
w1,v1 = eigen(AA_t)
U = v1
U
```




    array([[-0.41551283, -0.26918797, -0.50174522,  0.68293604,  0.19166903],
           [-0.33347608, -0.63069791, -0.19268409, -0.45313255, -0.49855549],
           [-0.60361532,  0.68471333, -0.28278996, -0.27655771, -0.10179322],
           [-0.14672854,  0.14916727,  0.44531083,  0.49724604, -0.71460794],
           [-0.57468432, -0.19665899,  0.6579154 , -0.06731614,  0.44008981]])




```python
w1[::-1].sort() #sorting the eigenvalues in descending order
w1 = np.sqrt(w1)
sig = np.diag(w1)
sig
```




    array([[18.79758132,  0.        ,  0.        ,  0.        ,  0.        ],
           [ 0.        , 10.07031422,  0.        ,  0.        ,  0.        ],
           [ 0.        ,  0.        ,  5.14396143,  0.        ,  0.        ],
           [ 0.        ,  0.        ,  0.        ,  4.28741257,  0.        ],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.63044602]])



### Calculating V / VT and Sigma


```python
A_tA = A_t.dot(A)
w2,v2 = eigen(A_tA)
V = v2
V
```




    array([[-0.25380113, -0.45529149, -0.77608085, -0.35215179, -0.04452269],
           [-0.42903133, -0.47119544,  0.06719816,  0.7490337 ,  0.1683446 ],
           [-0.40874866,  0.75341847, -0.43438937,  0.27657321,  0.00991348],
           [-0.56048312, -0.02715891,  0.33369138, -0.20410956, -0.72945899],
           [-0.51990413,  0.04803659,  0.30518659, -0.44360364,  0.6614145 ]])



Now V Transpose:


```python
V_t = np.transpose(V)
V_t
```




    array([[-0.25380113, -0.42903133, -0.40874866, -0.56048312, -0.51990413],
           [-0.45529149, -0.47119544,  0.75341847, -0.02715891,  0.04803659],
           [-0.77608085,  0.06719816, -0.43438937,  0.33369138,  0.30518659],
           [-0.35215179,  0.7490337 ,  0.27657321, -0.20410956, -0.44360364],
           [-0.04452269,  0.1683446 ,  0.00991348, -0.72945899,  0.6614145 ]])




```python
w2[::-1].sort() #sorting the eigenvalues in descending order
w2 = np.sqrt(w2)
sig2 = np.diag(w2)
sig2
```




    array([[18.79758132,  0.        ,  0.        ,  0.        ,  0.        ],
           [ 0.        , 10.07031422,  0.        ,  0.        ,  0.        ],
           [ 0.        ,  0.        ,  5.14396143,  0.        ,  0.        ],
           [ 0.        ,  0.        ,  0.        ,  4.28741257,  0.        ],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.63044602]])



We observe that the sigma we got from the AAt calculation (for U) is the same as the one from AtA calculation (for V)

### Function to calculate SVD

We can sum up the analysis and the calculation of U, Sigma, and V with the following function:


```python
def SVD (A):
    
    #calculating 'U' and 'Sigma'
    A_t = np.transpose(A)
    AA_t = A.dot(A_t)
    w1,v1 = eigen(AA_t)
#     for i in range(len(w1)):
#         if(w1[i]<10**-5):
#             w1[i]=0
#     w1 = np.sqrt(w1)
#     w1 = w1[w1!= 0]
#     sig = np.diag(w1)

    vec=[]
    for i in range(len(w1)):
        if(w1[i]<10**-5):
            w1[i]=0
        else:
            vec.append(i)
    w1 = np.sqrt(w1)
    w1 = w1[w1!=0]
    U = v1[:,vec]
    sig = np.diag(w1)
    
    #calculating 'V'
    A_tA = A_t.dot(A)
    w2,v2 = eigen(A_tA)
    vec2=[]
    for j in range(len(w2)):
        if(w2[j]<10**-5):
            w2[j]=0
        else:
            vec2.append(j)
    w2 = np.sqrt(w2)
    w2 = w2[w2!=0]
    V = v2[:,vec2]
    V_t = np.transpose(V)
    
    
    return U,sig,V_t
    
    
```

## Dimensionaltiy Reduction using SVD

### Energy

Energy of the eigenvalues of the sigma matrix refers to the sum of square of elements of the matrix. The idea is that if the energy after dropping one of the elements of sigma is greater than 90%, we can remove the eigenvector and the corresponding rows/columns from U and V.


```python
def energy(m,i):
    return 1-(m[i]/sum(np.square(m)))
```

**For Example:**


```python
energy([1,2,3],0)
```




    0.9285714285714286



We see that the 

### Applying SVD on the matrix from the book 

We will use a specific example given in the book (Mining Massive Datasets by Jure Leskovec). This way, we'll be able to observe and verify the application of dimensionality reduction using SVD.


```python
B = np.matrix([[1,1,1,0,0],[3,3,3,0,0],[4,4,4,0,0],[5,5,5,0,0],[0,0,0,4,4],[0,0,0,5,5],[0,0,0,2,2]])
B
```




    matrix([[1, 1, 1, 0, 0],
            [3, 3, 3, 0, 0],
            [4, 4, 4, 0, 0],
            [5, 5, 5, 0, 0],
            [0, 0, 0, 4, 4],
            [0, 0, 0, 5, 5],
            [0, 0, 0, 2, 2]])



Let's calculate it's rank, since we know that the number of non-zero eigenvalue this matrix would produce would be equal to it's rank.


```python
np.linalg.matrix_rank(B)
```




    2



Using the `SVD()` function defined above, we get 3 matrices corresponding to U, Sigma and V


```python
U,S,V_t = SVD(B)
U
```




    matrix([[0.14002801, 0.        ],
            [0.42008403, 0.        ],
            [0.56011203, 0.        ],
            [0.70014004, 0.        ],
            [0.        , 0.59628479],
            [0.        , 0.74535599],
            [0.        , 0.2981424 ]])




```python
S
```




    array([[12.36931688,  0.        ],
           [ 0.        ,  9.48683298]])




```python
V_t
```




    matrix([[0.57735027, 0.57735027, 0.57735027, 0.        , 0.        ],
            [0.        , 0.        , 0.        , 0.70710678, 0.70710678]])



Now, when we multiply U (Sigma) V_t, we should approximately get back the same matrix:


```python
U.dot(S.dot(V_t))
```




    matrix([[1., 1., 1., 0., 0.],
            [3., 3., 3., 0., 0.],
            [4., 4., 4., 0., 0.],
            [5., 5., 5., 0., 0.],
            [0., 0., 0., 4., 4.],
            [0., 0., 0., 5., 5.],
            [0., 0., 0., 2., 2.]])



If we had a matrix with a rank 3 and the categories of choices (of movie genres discussed in the book) were not clearly distinguishable, we can use dimensionality reduction to get a good approximation of those 2 categories.

### New Matrix:


```python
B2 = np.matrix([[1,1,1,0,0],[3,3,3,0,0],[4,4,4,0,0],[5,5,5,0,0],[0,2,0,4,4],[0,0,0,5,5],[0,1,0,2,2]])
B2
```




    matrix([[1, 1, 1, 0, 0],
            [3, 3, 3, 0, 0],
            [4, 4, 4, 0, 0],
            [5, 5, 5, 0, 0],
            [0, 2, 0, 4, 4],
            [0, 0, 0, 5, 5],
            [0, 1, 0, 2, 2]])




```python
U_prime,S_prime,V_t_prime = SVD(B2)
U_prime
```




    matrix([[-0.13759913+0.j,  0.02361145+0.j, -0.01080847+0.j],
            [-0.41279738+0.j,  0.07083435+0.j, -0.03242542+0.j],
            [-0.5503965 +0.j,  0.09444581+0.j, -0.04323389+0.j],
            [-0.68799563+0.j,  0.11805726+0.j, -0.05404236+0.j],
            [-0.15277509+0.j, -0.59110096+0.j,  0.65365084+0.j],
            [-0.07221651+0.j, -0.73131186+0.j, -0.67820922+0.j],
            [-0.07638754+0.j, -0.29555048+0.j,  0.32682542+0.j]])




```python
S_prime
```




    array([[12.48101469+0.j,  0.        +0.j,  0.        +0.j],
           [ 0.        +0.j,  9.50861406+0.j,  0.        +0.j],
           [ 0.        +0.j,  0.        +0.j,  1.34555971+0.j]])




```python
V_t_prime
```




    matrix([[ 0.56225841,  0.5928599 ,  0.56225841,  0.09013354,  0.09013354],
            [ 0.12664138, -0.02877058,  0.12664138, -0.69537622, -0.69537622],
            [-0.40966748,  0.80479152, -0.40966748, -0.0912571 , -0.0912571 ]])



Now, energy of S_prime with the 3rd eigenvalue of 1.3, is:


```python
energy(np.diag(S_prime),2)
```




    (0.994574355997+0j)



This is **much greater than 90%** and thus this eigenvalue can be removed from the matrix.


```python
Sigma_new = np.diag(np.diag(S_prime)[:2])
Sigma_new
```




    array([[12.48101469+0.j,  0.        +0.j],
           [ 0.        +0.j,  9.50861406+0.j]])



The U and V vectors would accordingly scale:


```python
U_new = U_prime[:,:2]
U_new
```




    matrix([[-0.13759913+0.j,  0.02361145+0.j],
            [-0.41279738+0.j,  0.07083435+0.j],
            [-0.5503965 +0.j,  0.09444581+0.j],
            [-0.68799563+0.j,  0.11805726+0.j],
            [-0.15277509+0.j, -0.59110096+0.j],
            [-0.07221651+0.j, -0.73131186+0.j],
            [-0.07638754+0.j, -0.29555048+0.j]])




```python
V_t_new = V_t_prime[:2,:]
V_t_new
```




    matrix([[ 0.56225841,  0.5928599 ,  0.56225841,  0.09013354,  0.09013354],
            [ 0.12664138, -0.02877058,  0.12664138, -0.69537622, -0.69537622]])



Now, the new multiplication of these terms should give a good estimation of the original matrix:


```python
M = U_new.dot(Sigma_new.dot(V_t_new))
M
```




    matrix([[-0.93717696+0.j, -1.02462313+0.j, -0.93717696+0.j,
             -0.31091367+0.j, -0.31091367+0.j],
            [-2.81153088+0.j, -3.0738694 +0.j, -2.81153088+0.j,
             -0.932741  +0.j, -0.932741  +0.j],
            [-3.74870783+0.j, -4.09849254+0.j, -3.74870783+0.j,
             -1.24365467+0.j, -1.24365467+0.j],
            [-4.68588479+0.j, -5.12311567+0.j, -4.68588479+0.j,
             -1.55456834+0.j, -1.55456834+0.j],
            [-1.78390197+0.j, -0.96875167+0.j, -1.78390197+0.j,
              3.7365319 +0.j,  3.7365319 +0.j],
            [-1.38741744+0.j, -0.3343018 +0.j, -1.38741744+0.j,
              4.75424033+0.j,  4.75424033+0.j],
            [-0.89195099+0.j, -0.48437583+0.j, -0.89195099+0.j,
              1.86826595+0.j,  1.86826595+0.j]])


