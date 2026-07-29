# Gauss-Jacobi Method

## Overview
The Gauss-Jacobi method is an iterative algorithm used to solve systems of linear equations. It is particularly useful for diagonally dominant systems and provides approximate solutions through successive approximations.

## How It Works
The method starts with an initial guess for the solution and iteratively improves it until convergence is reached. Each iteration updates all variables simultaneously using the values from the previous iteration.

## Algorithm
1. Rewrite each equation to isolate one variable
2. Start with an initial guess (usually all zeros)
3. Iterate: Update each variable using the previous iteration's values
4. Check for convergence (difference between iterations is below tolerance)
5. Repeat until convergence or maximum iterations reached

## Advantages
- Simple to implement
- Guaranteed convergence for diagonally dominant matrices
- Good for large sparse systems

## Limitations
- Slower convergence compared to Gauss-Seidel
- Requires the matrix to be diagonally dominant for guaranteed convergence

## Applications
- Solving large systems of linear equations
- Numerical analysis
- Engineering computations
- Scientific computing