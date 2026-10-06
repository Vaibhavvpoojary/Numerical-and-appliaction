def lagrange_interpolation(x_data, y_data, x):
    """
    Compute Lagrange interpolation at point x.
    
    Parameters:
    x_data : list of x-coordinates (unequal intervals allowed)
    y_data : list of y-coordinates
    x : point at which to evaluate the interpolation
    
    Returns:
    Interpolated value at x
    """
    n = len(x_data)
    if n != len(y_data):
        raise ValueError("x_data and y_data must have the same length")
    
    result = 0.0
    for i in range(n):
        # Compute Lagrange basis polynomial L_i(x)
        Li = 1.0
        for j in range(n):
            if i != j:
                Li *= (x - x_data[j]) / (x_data[i] - x_data[j])
        result += y_data[i] * Li
    
    return result


def lagrange_polynomial(x_data, y_data):
    """
    Return the Lagrange interpolation polynomial as a function.
    
    Parameters:
    x_data : list of x-coordinates
    y_data : list of y-coordinates
    
    Returns:
    Function that evaluates the interpolation polynomial
    """
    def polynomial(x):
        return lagrange_interpolation(x_data, y_data, x)
    return polynomial


def inverse_lagrange_interpolation(x_data, y_data, y):
    """
    Compute inverse Lagrange interpolation - find x for a given y.
    
    Parameters:
    x_data : list of x-coordinates
    y_data : list of y-coordinates
    y : y-value to find corresponding x
    
    Returns:
    Interpolated x value for given y
    """
    # Swap x and y for inverse interpolation
    return lagrange_interpolation(y_data, x_data, y)


def main():
    print("=" * 60)
    print("Lagrange Interpolation")
    print("=" * 60)
    
    n = int(input("\nEnter number of data points: "))
    
    print("\nEnter x values (unequal intervals allowed):")
    x_data = list(map(float, input().split()))
    
    print("\nEnter y values:")
    y_data = list(map(float, input().split()))
    
    if len(x_data) != n or len(y_data) != n:
        print("Error: Number of values doesn't match!")
        return
    
    poly = lagrange_polynomial(x_data, y_data)
    
    while True:
        print("\nOptions:")
        print("1. Evaluate at a point")
        print("2. Inverse interpolation (find x for given y)")
        print("3. Exit")
        choice = input("Enter choice (1/2/3): ")
        
        if choice == '1':
            x = float(input("Enter x to evaluate: "))
            y = poly(x)
            print(f"f({x}) = {y:.6f}")
        elif choice == '2':
            y = float(input("Enter y to find x: "))
            x = inverse_lagrange_interpolation(x_data, y_data, y)
            print(f"x for y={y} is approximately {x:.6f}")
        elif choice == '3':
            break
        else:
            print("Invalid choice!")


if __name__ == "__main__":
    main()