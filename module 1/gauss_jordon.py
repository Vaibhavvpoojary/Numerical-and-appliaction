# Gauss-Jordan Elimination Method

n = int(input("Enter the number of variables: "))

# Input augmented matrix
a = []
print("Enter the augmented matrix row-wise:")
for i in range(n):
    row = list(map(float, input().split()))
    a.append(row)

# Gauss-Jordan Elimination
for i in range(n):
    # Check for zero pivot
    if a[i][i] == 0:
        print("Mathematical Error! Zero pivot encountered.")
        exit()

    # Make pivot element 1
    pivot = a[i][i]
    for j in range(n + 1):
        a[i][j] /= pivot

    # Make all other elements in the pivot column 0
    for k in range(n):
        if k != i:
            factor = a[k][i]
            for j in range(n + 1):
                a[k][j] -= factor * a[i][j]

# Display solution
print("\nSolution:")
for i in range(n):
    print(f"x{i + 1} = {a[i][n]:.2f}")