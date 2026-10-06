def cubic_spline(x_data, y_data):
    """
    Compute natural cubic spline interpolation.
    
    Parameters:
    x_data : list of x-coordinates (must be strictly increasing)
    y_data : list of y-coordinates
    
    Returns:
    Tuple of (a, b, c, d) coefficients for each interval
    Each interval i has: S_i(x) = a[i] + b[i]*(x-x_i) + c[i]*(x-x_i)^2 + d[i]*(x-x_i)^3
    """
    n = len(x_data)
    if n != len(y_data):
        raise ValueError("x_data and y_data must have the same length")
    if n < 2:
        raise ValueError("At least 2 data points required")
    
    # Check if x_data is strictly increasing
    for i in range(1, n):
        if x_data[i] <= x_data[i-1]:
            raise ValueError("x_data must be strictly increasing")
    
    # Step 1: Compute h[i] = x[i+1] - x[i]
    h = [x_data[i+1] - x_data[i] for i in range(n-1)]
    
    # Step 2: Set up tridiagonal system for second derivatives (c coefficients)
    # Natural spline: c[0] = c[n-1] = 0
    # System: h[i-1]*c[i-1] + 2*(h[i-1]+h[i])*c[i] + h[i]*c[i+1] = 3*((y[i+1]-y[i])/h[i] - (y[i]-y[i-1])/h[i-1])
    
    if n == 2:
        # Linear spline for 2 points
        a = [y_data[0], y_data[1]]
        b = [(y_data[1] - y_data[0]) / h[0], 0]
        c = [0, 0]
        d = [0, 0]
        return a, b, c, d
    
    # Coefficients for tridiagonal matrix
    A = [0.0] * (n - 2)  # sub-diagonal (h[i-1])
    B = [0.0] * (n - 2)  # diagonal (2*(h[i-1] + h[i]))
    C = [0.0] * (n - 2)  # super-diagonal (h[i])
    D = [0.0] * (n - 2)  # right-hand side
    
    for i in range(1, n-1):
        A[i-1] = h[i-1]
        B[i-1] = 2 * (h[i-1] + h[i])
        C[i-1] = h[i]
        D[i-1] = 3 * ((y_data[i+1] - y_data[i]) / h[i] - (y_data[i] - y_data[i-1]) / h[i-1])
    
    # Solve tridiagonal system using Thomas algorithm
    # Forward elimination
    for i in range(1, n-2):
        factor = A[i-1] / B[i-2]
        B[i-1] -= factor * C[i-2]
        D[i-1] -= factor * D[i-2]
    
    # Back substitution
    c_internal = [0.0] * (n - 2)
    c_internal[-1] = D[-1] / B[-1]
    for i in range(n-4, -1, -1):
        c_internal[i] = (D[i] - C[i] * c_internal[i+1]) / B[i]
    
    # Full c array with natural boundary conditions
    c = [0.0] + c_internal + [0.0]
    
    # Step 3: Compute a, b, d coefficients
    a = y_data[:]
    b = [0.0] * (n-1)
    d = [0.0] * (n-1)
    
    for i in range(n-1):
        b[i] = (y_data[i+1] - y_data[i]) / h[i] - h[i] * (2*c[i] + c[i+1]) / 3
        d[i] = (c[i+1] - c[i]) / (3 * h[i])
    
    return a, b, c, d


def evaluate_cubic_spline(x_data, coeffs, x):
    """
    Evaluate cubic spline at point x.
    
    Parameters:
    x_data : list of x-coordinates (knots)
    coeffs : tuple (a, b, c, d) from cubic_spline()
    x : point to evaluate
    
    Returns:
    Interpolated value at x
    """
    a, b, c, d = coeffs
    n = len(x_data)
    
    # Find interval
    if x <= x_data[0]:
        i = 0
    elif x >= x_data[-1]:
        i = n - 2
    else:
        for i in range(n-1):
            if x_data[i] <= x <= x_data[i+1]:
                break
    
    dx = x - x_data[i]
    return a[i] + b[i]*dx + c[i]*dx**2 + d[i]*dx**3


def cubic_spline_function(x_data, y_data):
    """
    Return a function that evaluates the cubic spline interpolation.
    """
    coeffs = cubic_spline(x_data, y_data)
    def spline(x):
        return evaluate_cubic_spline(x_data, coeffs, x)
    return spline


def main():
    print("=" * 60)
    print("Natural Cubic Spline Interpolation")
    print("=" * 60)
    
    n = int(input("\nEnter number of data points (minimum 2): "))
    
    print("\nEnter x values (strictly increasing):")
    x_data = list(map(float, input().split()))
    
    print("\nEnter y values:")
    y_data = list(map(float, input().split()))
    
    if len(x_data) != n or len(y_data) != n:
        print("Error: Number of values doesn't match!")
        return
    
    try:
        coeffs = cubic_spline(x_data, y_data)
        a, b, c, d = coeffs
        
        print("\nSpline Coefficients:")
        print("-" * 60)
        print(f"{'Interval':<12} {'a':<12} {'b':<12} {'c':<12} {'d':<12}")
        print("-" * 60)
        for i in range(n-1):
            print(f"[{x_data[i]}, {x_data[i+1]}]  {a[i]:.6f}  {b[i]:.6f}  {c[i]:.6f}  {d[i]:.6f}")
        
        spline = cubic_spline_function(x_data, y_data)
        
        while True:
            print("\nOptions:")
            print("1. Evaluate at a point")
            print("2. Evaluate at multiple points")
            print("3. Exit")
            choice = input("Enter choice (1/2/3): ")
            
            if choice == '1':
                x = float(input("Enter x to evaluate: "))
                y = spline(x)
                print(f"S({x}) = {y:.6f}")
            elif choice == '2':
                points = list(map(float, input("Enter x values (space-separated): ").split()))
                print(f"\n{'x':<12} {'S(x)':<12}")
                print("-" * 24)
                for x in points:
                    print(f"{x:<12.6f} {spline(x):<12.6f}")
            elif choice == '3':
                break
            else:
                print("Invalid choice!")
                
    except ValueError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()