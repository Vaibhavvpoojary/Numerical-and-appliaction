# Gauss-Seidel Method

n = int(input("Enter the number of variables: "))

# Input coefficient matrix
A = []
print("Enter the coefficient matrix:")
for i in range(n):
    row = list(map(float, input().split()))
    A.append(row)

# Input constant matrix
B = list(map(float, input("Enter the constant terms: ").split()))

# Initial guesses
X = [0.0] * n

# Input tolerance and maximum iterations
tolerance = float(input("Enter tolerance (e.g., 0.0001): "))
max_iter = int(input("Enter maximum number of iterations: "))

print("\nIterations:")

for iteration in range(max_iter):
    X_old = X.copy()

    for i in range(n):
        sum1 = 0
        for j in range(n):
            if j != i:
                sum1 += A[i][j] * X[j]

        X[i] = (B[i] - sum1) / A[i][i]

    print(f"Iteration {iteration + 1}: ", end="")
    for i in range(n):
        print(f"x{i+1} = {X[i]:.6f}", end="  ")
    print()

    # Check convergence
    error = max(abs(X[i] - X_old[i]) for i in range(n))
    if error < tolerance:
        print("\nSolution converged.")
        break
else:
    print("\nMaximum iterations reached.")

# Display final solution
print("\nFinal Solution:")
for i in range(n):
    print(f"x{i+1} = {X[i]:.6f}")