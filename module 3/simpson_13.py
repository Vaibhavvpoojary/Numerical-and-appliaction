def simpson_13(f, a, b, n):
    """
    Simpson's 1/3 rule for numerical integration.
    
    Parameters:
    f : function to integrate
    a : lower limit
    b : upper limit
    n : number of subintervals (must be even)
    
    Returns:
    Approximate integral value
    """
    if n % 2 != 0:
        raise ValueError("Number of subintervals n must be even for Simpson's 1/3 rule")
    
    h = (b - a) / n
    x = [a + i * h for i in range(n + 1)]
    y = [f(xi) for xi in x]
    
    # Simpson's 1/3 formula: (h/3) * [y0 + yn + 4*(y1+y3+...) + 2*(y2+y4+...)]
    integral = y[0] + y[n]
    
    for i in range(1, n):
        if i % 2 == 1:  # odd indices
            integral += 4 * y[i]
        else:  # even indices
            integral += 2 * y[i]
    
    integral *= h / 3
    return integral


def simpson_13_tabulated(x_data, y_data):
    """
    Simpson's 1/3 rule for tabulated data.
    
    Parameters:
    x_data : list of x-coordinates (equally spaced)
    y_data : list of y-coordinates
    
    Returns:
    Approximate integral value
    """
    n = len(x_data) - 1
    if n % 2 != 0:
        raise ValueError("Number of subintervals must be even (need odd number of points)")
    
    # Check equal spacing
    h = x_data[1] - x_data[0]
    for i in range(2, len(x_data)):
        if abs(x_data[i] - x_data[i-1] - h) > 1e-10:
            raise ValueError("x_data must be equally spaced")
    
    integral = y_data[0] + y_data[n]
    
    for i in range(1, n):
        if i % 2 == 1:
            integral += 4 * y_data[i]
        else:
            integral += 2 * y_data[i]
    
    integral *= h / 3
    return integral


def simpson_13_double(f, ax, bx, ay, by, nx, ny):
    """
    Simpson's 1/3 rule for double integrals.
    
    Parameters:
    f : function of two variables f(x, y)
    ax, bx : x limits
    ay, by : y limits
    nx, ny : number of subintervals in x and y (must be even)
    
    Returns:
    Approximate double integral value
    """
    if nx % 2 != 0 or ny % 2 != 0:
        raise ValueError("nx and ny must be even for Simpson's 1/3 rule")
    
    hx = (bx - ax) / nx
    hy = (by - ay) / ny
    
    x = [ax + i * hx for i in range(nx + 1)]
    y = [ay + j * hy for j in range(ny + 1)]
    
    # Compute function values at grid points
    z = [[f(xi, yj) for yj in y] for xi in x]
    
    # Apply Simpson's rule in y-direction for each x
    y_integrals = []
    for i in range(nx + 1):
        integral_y = z[i][0] + z[i][ny]
        for j in range(1, ny):
            if j % 2 == 1:
                integral_y += 4 * z[i][j]
            else:
                integral_y += 2 * z[i][j]
        integral_y *= hy / 3
        y_integrals.append(integral_y)
    
    # Apply Simpson's rule in x-direction
    integral = y_integrals[0] + y_integrals[nx]
    for i in range(1, nx):
        if i % 2 == 1:
            integral += 4 * y_integrals[i]
        else:
            integral += 2 * y_integrals[i]
    
    integral *= hx / 3
    return integral


def main():
    print("=" * 60)
    print("Simpson's 1/3 Rule for Numerical Integration")
    print("=" * 60)
    
    while True:
        print("\nOptions:")
        print("1. Single integral (function)")
        print("2. Single integral (tabulated data)")
        print("3. Double integral (function)")
        print("4. Exit")
        choice = input("Enter choice (1-4): ")
        
        if choice == '1':
            expr = input("Enter function f(x) (e.g., 'x**2', 'math.sin(x)'): ")
            f = lambda x: eval(expr, {"math": __import__("math"), "x": x})
            
            a = float(input("Enter lower limit a: "))
            b = float(input("Enter upper limit b: "))
            n = int(input("Enter number of subintervals (even): "))
            
            try:
                result = simpson_13(f, a, b, n)
                print(f"\nIntegral = {result:.8f}")
            except ValueError as e:
                print(f"Error: {e}")
                
        elif choice == '2':
            n = int(input("Enter number of data points (odd): "))
            
            print("Enter x values (equally spaced):")
            x_data = list(map(float, input().split()))
            
            print("Enter y values:")
            y_data = list(map(float, input().split()))
            
            if len(x_data) != n or len(y_data) != n:
                print("Error: Number of values doesn't match!")
                continue
            
            try:
                result = simpson_13_tabulated(x_data, y_data)
                print(f"\nIntegral = {result:.8f}")
            except ValueError as e:
                print(f"Error: {e}")
                
        elif choice == '3':
            expr = input("Enter function f(x,y) (e.g., 'x*y', 'math.sin(x)*math.cos(y)'): ")
            f = lambda x, y: eval(expr, {"math": __import__("math"), "x": x, "y": y})
            
            ax = float(input("Enter x lower limit: "))
            bx = float(input("Enter x upper limit: "))
            ay = float(input("Enter y lower limit: "))
            by = float(input("Enter y upper limit: "))
            nx = int(input("Enter nx (even): "))
            ny = int(input("Enter ny (even): "))
            
            try:
                result = simpson_13_double(f, ax, bx, ay, by, nx, ny)
                print(f"\nDouble integral = {result:.8f}")
            except ValueError as e:
                print(f"Error: {e}")
                
        elif choice == '4':
            break
        else:
            print("Invalid choice!")


if __name__ == "__main__":
    main()