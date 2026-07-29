# Gauss-Jacobi Method

## 📌 Overview
This project implements the **Gauss-Jacobi Iteration Method** in Python to solve systems of linear equations. The program uses an iterative approach to find approximate solutions by repeatedly updating all variables simultaneously until convergence is achieved.

## 🚀 Features
- Solves a system of **n linear equations** with **n variables**
- Accepts user input for the coefficient matrix and right-hand side vector
- Optional initial guess for faster convergence
- Configurable tolerance and maximum iterations
- Detects and warns about non-diagonally dominant matrices
- Displays iteration progress and convergence information

## 🛠️ Technologies Used
- Python 3

## 📂 Project Structure
```
gauss_jacobi/
│── gauss_jacobi.py
│── readme.md
```

## ▶️ How to Run

1. Navigate to the gauss_jacobi folder
2. Run the program:
   ```bash
   python gauss_jacobi.py
   ```
3. Enter the number of variables
4. Enter the coefficient matrix row by row
5. Enter the right-hand side vector
6. Optionally provide an initial guess
7. Set tolerance and maximum iterations (or use defaults)

## 💻 Example

### Input
```
Enter the number of variables: 3
Enter the coefficient matrix (3x3):
Row 1: 4 -1 0
Row 2: -1 4 -1
Row 3: 0 -1 4
Enter the right-hand side vector (3 values):
3 9 18
```

### Output
```
Gauss-Jacobi Iteration:
------------------------------------------------------------
Iteration 1: [0.75, 2.25, 4.5]
Iteration 2: [0.9375, 2.8125, 5.53125]
...
Converged after 15 iterations

SOLUTION:
============================================================
x1 = 1.000000
x2 = 3.000000
x3 = 5.000000
============================================================
```

## 📖 Algorithm
1. Rewrite each equation to isolate one variable: x_i = (b_i - Σ(a_ij * x_j)) / a_ii
2. Initialize solution vector (zeros or user-provided guess)
3. Iterate: Update all variables simultaneously using previous iteration values
4. Check for convergence: ||x_new - x_old|| < tolerance
5. Repeat until convergence or maximum iterations reached

## ⏱️ Time Complexity
- **Time Complexity:** O(n² × iterations)
- **Space Complexity:** O(n²)

## 🎯 Applications
- Solving large systems of linear equations
- Numerical analysis and computational mathematics
- Engineering computations (heat transfer, fluid dynamics)
- Scientific computing and simulations
- Educational purposes for learning iterative methods

## 📝 Notes
- The method guarantees convergence for strictly diagonally dominant matrices
- Convergence may be slow for some systems
- Gauss-Seidel method typically converges faster as it uses updated values immediately
- For best results, ensure the coefficient matrix is diagonally dominant

## 📜 License
This project is open-source and available for educational and personal use.