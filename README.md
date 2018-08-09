# Nested Optimization Scheme

## Overview

This program computes global extrema (i.e. minima and maxima) of Lipschitz continuous functions. Input functions can be one-dimensional or multidimensional. A Lipschitz constant can be dynamically estimated (i.e. it can be updated in the course of the algorithm). Hence, a Lipschitz constant need not be known in advance. 

The optimization algorithm for one-dimensional Lipschitz continuous functions used in this program is known as the sequential global optimization method. This is detailed in *A Sequential Method Seeking the Global Maximum of a Function* by Bruno O. Schubert. 

As for multidimensional Lipschitz continuous functions, the nested optimization scheme is used. In this scheme, a global optimization problem for a multidimensional function is reduced to a collection of one-dimensional optimization subproblems that form a nested relationship (hence the name "nested"). To solve each of these one-dimensional optimization subproblems, the sequential method is used. Further, a Lipschitz constant can be dynamically estimated, in which case the algorithm is called the "adaptive" nested optimization scheme. The adaptive nested scheme is explained in *Local Tuning in Nested Scheme of Global Optimization* by Victor Gergel, Vladimir Grishagin, and Ruslan Israfilov. 