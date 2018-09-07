# BDA492 - Data Mining and its Applications 
## _Assignment 1_

This repository contains MATLAB scripts for the prompt `Assignment1.pdf`

There are 3 sections of the script:
1. **Question 1: ** We plot the following implicit function for the p-norm of a unit-ball in the R2 domain, using the `pnorm1()` function: _pnorm1(v1,v2,p)-1 = 0_.  For plotting we use the `fimplicit()` function. We plot this function for the values of p as 0.5,1,2,4,1000 (infinity). What we observe is (in general), for values of 0<p<1, we get a convex shaped curve. As we increase p, we see that for 1, we get a rhombus-shaped plot, circle for 2 and a square for infinity (1000). A conclusion derived from these plots is that n-norm is always greater than n-1,n-2…1 norms.
2. **Question 2:** We plot the mean-square error of linear regression, i.e. 
(img)
	Using the `fimplicit3()` function, and the above formula as our implicit function, we get a 3D plot with `m` on the x-axis, `c` on the y-axis and error `e` on the z-axis. The plot thus obtained is observed to be a bell-shaped surface.

	Similarly for the 2D plot, using the `fimplicit()` function, we obtain a graph with `m` on the x-axis and error `e` on the y-axis. We observe the corresponding graph to be a parabolic function.
3. **Question 3:**  For the values of x1 and y1 in the previous question, We make a cost function `costfunction()` to calculate the error/cost function for linear regression i.e. `J = 1/(2*m) * sum((h-y).^2)`.  We then use the `gradient_descent()`  function to optimize the cost function J to get the values of theta (coefficients of the linear equations).
