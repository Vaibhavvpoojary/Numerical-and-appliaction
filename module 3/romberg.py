def trapezoidal(f, a, b, n):
    """Composite trapezoidal rule."""
    h = (b - a) / n
    x = [a + i * h for i in range(n + 1)]
    y = [f(xi) for xi in x]
    
    integral = y[0] + y[n]
    for i in range(1, n):
        integral += 2 * y[i]
    
    return integral * h / 2


def romberg(f, a, b, max_level=5, tolerance=1e-10):
    """
    Romberg integration method.
    
    Parameters:
    f : function to integrate
    a : lower limit
    b : upper limit
    max_level : maximum extrapolation level (default 5)
    tolerance : convergence tolerance (default 1e-10)
    
    Returns:
    (integral_value, R_table, levels_used)
    """
    R = []
    
    # Level 0: composite trapezoidal with 1 interval
    R.append([trapezoidal(f, a, b, 1)])
    
    for k in range(1, max_level + 1):
        # Level k: composite trapezoidal with 2^k intervals
        h = (b - a) / (2**k)
        # Compute new midpoint evaluations only
        mid_sum = sum(f(a + (i - 0.5) * 2 * h) for i in range(1, 2**(k-1) + 1))
        Rk_0 = 0.5 * R[k-1][0] + h * mid_sum
        
        # Richardson extrapolation
        Rk = [Rk_0]
        for j in range(1, k + 1):
            factor = 4**j
            Rk_j = (factor * Rk[j-1] - R[k-1][j-1]) / (factor - 1)
            Rk.append(Rk_j)
        
        R.append(Rk)
        
        # Check convergence
        if k >= 1 and abs(R[k][k] - R[k-1][k-1]) < tolerance:
            return R[k][k], R, k + 1
    
    return R[max_level][max_level], R, max_level + 1


def romberg_table(R):
    """Print Romberg table in a formatted way."""
    print("\nRomberg Table:")
    print("-" * 80)
    header = f"{'k\\j':>6}"
    for j in range(len(R)):
        header += f"  R[{j}]:>16"
    print(header)
    print("-" * 80)
    
    for k in range(len(R)):
        row = f"  {k}: "
        for j in range(k + 1):
            row += f"  {R[k][j]:16.10f}"
        print(row)
    print("-" * 80)


def romberg_double(f, ax, bx, ay, by, max_level=4, tolerance=1e-8):
    """
    Romberg integration for double integrals.
    Uses Romberg in x-direction, with inner integral in y using Romberg.
    """
    def inner_integral(x):
        # For each x, integrate f(x,y) in y using Romberg
        result, _, _ = romberg(lambda y: f(x, y), ay, by, max_level, tolerance)
        return result
    
    # Now integrate the inner integral in x using Romberg
    return romberg(inner_integral, ax, bx, max_level, tolerance)


def main():
    print("=" * 60)
    print("Romberg Integration Method")
    print("=" * 60)
    
    while True:
        print("\nOptions:")
        print("1. Single integral")
        print("2. Double integral")
        print("3. Show Romberg table for single integral")
        print("4. Exit")
        choice = input("Enter choice (1-4): ")
        
        if choice == '1':
            expr = input("Enter function f(x) (e.g., 'x**2', 'math.sin(x)'): ")
            f = lambda x: eval(expr, {"math": __import__("math"), "x": x})
            
            a = float(input("Enter lower limit a: "))
            b = float(input("Enter upper limit b: "))
            max_level = int(input("Enter max level (default 5): ") or "5")
            tolerance = float(input("Enter tolerance (default 1e-10): ") or "1e-10")
            
            result, R, levels = romberg(f, a, b, max_level, tolerance)
            print(f"\nIntegral = {result:.12f}")
            print(f"Levels used: {levels}")
            print(f"Estimated error: {abs(R[levels-1][levels-1] - R[levels-2][levels-2]) if levels > 1 else 'N/A':.2e}")
            
        elif choice == '2':
            expr = input("Enter function f(x,y) (e.g., 'x*y', 'math.sin(x)*math.cos(y)'): ")
            f = lambda x, y: eval(expr, {"math": __import__("math"), "x": x, "y": y})
            
            ax = float(input("Enter x lower limit: "))
            bx = float(input("Enter x upper limit: "))
            ay = float(input("Enter y lower limit: "))
            by = float(input("Enter y upper limit: "))
            max_level = int(input("Enter max level (default 4): ") or "4")
            tolerance = float(input("Enter tolerance (default 1e-8): ") or "1e-8")
            
            result, R, levels = romberg_double(f, ax, bx, ay, by, max_level, tolerance)
            print(f"\nDouble integral = {result:.12f}")
            print(f"Levels used: {levels}")
            
        elif choice == '3':
            expr = input("Enter function f(x) (e.g., 'x**2', 'math.sin(x)'): ")
            f = lambda x: eval(expr, {"math": __import__("math"), "x": x})
            
            a = float(input("Enter lower limit a: "))
            b = float(input("Enter upper limit b: "))
            max_level = int(input("Enter max level (default 5): ") or "5")
            
            result, R, levels = romberg(f, a, b, max_level)
            print(f"\nIntegral = {result:.12f}")
            romberg_table(R)
            
        elif choice == '4':
            break
        else:
            print("Invalid choice!")


if __name__ == "__main__":
    main()