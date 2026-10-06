def gauss_jacobi(A, b, initial_guess=None, tolerance=1e-10, max_iterations=1000):
    """
    Solve a system of linear equations using Gauss-Jacobi iteration method.
    
    Parameters:
    A : Coefficient matrix (n x n)
    b : Right-hand side vector (n x 1)
    initial_guess : Initial guess for solution (optional)
    tolerance : Convergence tolerance
    max_iterations : Maximum number of iterations
    
    Returns:
    x : Solution vector
    iterations : Number of iterations performed
    """
    n = len(A)
    
    # Initialize with zeros if no initial guess provided
    if initial_guess is None:
        x = [0.0] * n
    else:
        x = initial_guess[:]
    
    # Check if matrix is diagonally dominant
    for i in range(n):
        diagonal = abs(A[i][i])
        row_sum = sum(abs(A[i][j]) for j in range(n) if j != i)
        if diagonal <= row_sum:
            print(f"Warning: Matrix is not strictly diagonally dominant at row {i+1}")
            print(f"Diagonal element: {diagonal}, Sum of other elements: {row_sum}")
    
    print("\nGauss-Jacobi Iteration:")
    print("-" * 60)
    
    for iteration in range(max_iterations):
        x_old = x[:]
        
        # Update each variable
        for i in range(n):
            sum_val = sum(A[i][j] * x_old[j] for j in range(n) if j != i)
            x[i] = (b[i] - sum_val) / A[i][i]
        
        # Print iteration details
        print(f"Iteration {iteration + 1}: {[round(val, 6) for val in x]}")
        
        # Check for convergence
        max_diff = max(abs(x[i] - x_old[i]) for i in range(n))
        if max_diff < tolerance:
            print("-" * 60)
            print(f"Converged after {iteration + 1} iterations")
            return x, iteration + 1
    
    print("-" * 60)
    print(f"Maximum iterations ({max_iterations}) reached")
    return x, max_iterations


def main():
    print("=" * 60)
    print("Gauss-Jacobi Method Solver")
    print("=" * 60)
    
    # Get number of variables
    n = int(input("\nEnter the number of variables: "))
    
    # Get coefficient matrix
    print(f"\nEnter the coefficient matrix ({n}x{n}):")
    A = []
    for i in range(n):
        row = list(map(float, input(f"Row {i+1}: ").split()))
        A.append(row)
    
    # Get right-hand side vector
    print(f"\nEnter the right-hand side vector ({n} values):")
    b = list(map(float, input().split()))
    
    # Get initial guess (optional)
    use_initial = input("\nDo you want to provide an initial guess? (y/n): ").lower()
    if use_initial == 'y':
        print(f"Enter initial guess ({n} values):")
        initial_guess = list(map(float, input().split()))
    else:
        initial_guess = None
    
    # Get tolerance
    tolerance = float(input("\nEnter tolerance (default 1e-10): ") or "1e-10")
    
    # Get max iterations
    max_iter = int(input("Enter maximum iterations (default 1000): ") or "1000")
    
    # Solve the system
    print("\n" + "=" * 60)
    solution, iterations = gauss_jacobi(A, b, initial_guess, tolerance, max_iterations=max_iter)
    
    # Display results
    print("\n" + "=" * 60)
    print("SOLUTION:")
    print("=" * 60)
    for i, val in enumerate(solution):
        print(f"x{i+1} = {val:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()