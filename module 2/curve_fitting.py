import math


def polynomial_fit(x_data, y_data, degree):
    """
    Polynomial curve fitting using least squares method.
    
    Parameters:
    x_data : list of x-coordinates
    y_data : list of y-coordinates
    degree : degree of polynomial
    
    Returns:
    coefficients : list of coefficients [a_0, a_1, ..., a_n] for a_0 + a_1*x + ... + a_n*x^n
    """
    n = len(x_data)
    if n != len(y_data):
        raise ValueError("x_data and y_data must have the same length")
    if n <= degree:
        raise ValueError("Need more data points than degree")
    
    # Build normal equations: A^T * A * c = A^T * y
    # A is Vandermonde matrix
    m = degree + 1
    
    # Compute A^T * A (symmetric matrix)
    ATA = [[0.0] * m for _ in range(m)]
    for i in range(m):
        for j in range(m):
            ATA[i][j] = sum(x_data[k]**(i+j) for k in range(n))
    
    # Compute A^T * y
    ATy = [0.0] * m
    for i in range(m):
        ATy[i] = sum(x_data[k]**i * y_data[k] for k in range(n))
    
    # Solve linear system ATA * c = ATy using Gaussian elimination
    # Augmented matrix
    aug = [ATA[i] + [ATy[i]] for i in range(m)]
    
    # Forward elimination
    for i in range(m):
        # Find pivot
        max_row = i
        for k in range(i+1, m):
            if abs(aug[k][i]) > abs(aug[max_row][i]):
                max_row = k
        aug[i], aug[max_row] = aug[max_row], aug[i]
        
        if abs(aug[i][i]) < 1e-12:
            raise ValueError("Matrix is singular or nearly singular")
        
        # Normalize pivot row
        pivot = aug[i][i]
        for j in range(i, m+1):
            aug[i][j] /= pivot
        
        # Eliminate below
        for k in range(i+1, m):
            factor = aug[k][i]
            for j in range(i, m+1):
                aug[k][j] -= factor * aug[i][j]
    
    # Back substitution
    coeffs = [0.0] * m
    for i in range(m-1, -1, -1):
        coeffs[i] = aug[i][m]
        for j in range(i+1, m):
            coeffs[i] -= aug[i][j] * coeffs[j]
    
    return coeffs


def evaluate_polynomial(coeffs, x):
    """Evaluate polynomial at x using Horner's method."""
    result = 0.0
    for c in reversed(coeffs):
        result = result * x + c
    return result


def polynomial_function(coeffs):
    """Return a function that evaluates the polynomial."""
    def poly(x):
        return evaluate_polynomial(coeffs, x)
    return poly


def exponential_fit(x_data, y_data):
    """
    Exponential curve fitting: y = a * exp(b * x)
    Linearized: ln(y) = ln(a) + b*x
    """
    # Check for positive y values
    if any(y <= 0 for y in y_data):
        raise ValueError("All y values must be positive for exponential fit")
    
    ln_y = [math.log(y) for y in y_data]
    coeffs = polynomial_fit(x_data, ln_y, 1)
    a = math.exp(coeffs[0])
    b = coeffs[1]
    return a, b


def power_fit(x_data, y_data):
    """
    Power curve fitting: y = a * x^b
    Linearized: ln(y) = ln(a) + b*ln(x)
    """
    if any(x <= 0 for x in x_data) or any(y <= 0 for y in y_data):
        raise ValueError("All x and y values must be positive for power fit")
    
    ln_x = [math.log(x) for x in x_data]
    ln_y = [math.log(y) for y in y_data]
    coeffs = polynomial_fit(ln_x, ln_y, 1)
    a = math.exp(coeffs[0])
    b = coeffs[1]
    return a, b


def logarithmic_fit(x_data, y_data):
    """
    Logarithmic curve fitting: y = a + b*ln(x)
    """
    if any(x <= 0 for x in x_data):
        raise ValueError("All x values must be positive for logarithmic fit")
    
    ln_x = [math.log(x) for x in x_data]
    coeffs = polynomial_fit(ln_x, y_data, 1)
    return coeffs[0], coeffs[1]


def r_squared(x_data, y_data, model_func):
    """Calculate R-squared (coefficient of determination)."""
    y_mean = sum(y_data) / len(y_data)
    ss_tot = sum((y - y_mean)**2 for y in y_data)
    ss_res = sum((y - model_func(x))**2 for x, y in zip(x_data, y_data))
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0


def main():
    print("=" * 60)
    print("Curve Fitting (Least Squares Method)")
    print("=" * 60)
    
    n = int(input("\nEnter number of data points: "))
    
    print("\nEnter x values:")
    x_data = list(map(float, input().split()))
    
    print("\nEnter y values:")
    y_data = list(map(float, input().split()))
    
    if len(x_data) != n or len(y_data) != n:
        print("Error: Number of values doesn't match!")
        return
    
    while True:
        print("\nCurve Fitting Options:")
        print("1. Polynomial fit")
        print("2. Exponential fit (y = a*e^(bx))")
        print("3. Power fit (y = a*x^b)")
        print("4. Logarithmic fit (y = a + b*ln(x))")
        print("5. Compare all fits")
        print("6. Exit")
        choice = input("Enter choice (1-6): ")
        
        if choice == '1':
            degree = int(input("Enter polynomial degree: "))
            try:
                coeffs = polynomial_fit(x_data, y_data, degree)
                poly = polynomial_function(coeffs)
                r2 = r_squared(x_data, y_data, poly)
                
                print(f"\nPolynomial coefficients (a_0 + a_1*x + ... + a_{degree}*x^{degree}):")
                for i, c in enumerate(coeffs):
                    print(f"  a_{i} = {c:.6f}")
                print(f"R-squared = {r2:.6f}")
                
                # Evaluate
                while True:
                    x = input("\nEnter x to evaluate (or 'back' to return): ")
                    if x.lower() == 'back':
                        break
                    try:
                        x_val = float(x)
                        print(f"y({x_val}) = {poly(x_val):.6f}")
                    except ValueError:
                        print("Invalid input!")
                        
            except ValueError as e:
                print(f"Error: {e}")
                
        elif choice == '2':
            try:
                a, b = exponential_fit(x_data, y_data)
                model = lambda x: a * math.exp(b * x)
                r2 = r_squared(x_data, y_data, model)
                
                print(f"\nExponential fit: y = {a:.6f} * e^({b:.6f} * x)")
                print(f"R-squared = {r2:.6f}")
                
                while True:
                    x = input("\nEnter x to evaluate (or 'back' to return): ")
                    if x.lower() == 'back':
                        break
                    try:
                        x_val = float(x)
                        print(f"y({x_val}) = {model(x_val):.6f}")
                    except ValueError:
                        print("Invalid input!")
            except ValueError as e:
                print(f"Error: {e}")
                
        elif choice == '3':
            try:
                a, b = power_fit(x_data, y_data)
                model = lambda x: a * (x ** b)
                r2 = r_squared(x_data, y_data, model)
                
                print(f"\nPower fit: y = {a:.6f} * x^{b:.6f}")
                print(f"R-squared = {r2:.6f}")
                
                while True:
                    x = input("\nEnter x to evaluate (or 'back' to return): ")
                    if x.lower() == 'back':
                        break
                    try:
                        x_val = float(x)
                        print(f"y({x_val}) = {model(x_val):.6f}")
                    except ValueError:
                        print("Invalid input!")
            except ValueError as e:
                print(f"Error: {e}")
                
        elif choice == '4':
            try:
                a, b = logarithmic_fit(x_data, y_data)
                model = lambda x: a + b * math.log(x)
                r2 = r_squared(x_data, y_data, model)
                
                print(f"\nLogarithmic fit: y = {a:.6f} + {b:.6f} * ln(x)")
                print(f"R-squared = {r2:.6f}")
                
                while True:
                    x = input("\nEnter x to evaluate (or 'back' to return): ")
                    if x.lower() == 'back':
                        break
                    try:
                        x_val = float(x)
                        print(f"y({x_val}) = {model(x_val):.6f}")
                    except ValueError:
                        print("Invalid input!")
            except ValueError as e:
                print(f"Error: {e}")
                
        elif choice == '5':
            print("\n" + "=" * 60)
            print("Model Comparison")
            print("=" * 60)
            print(f"{'Model':<25} {'R-squared':<12}")
            print("-" * 60)
            
            # Polynomial (degree 1, 2, 3)
            for deg in [1, 2, 3]:
                if n > deg:
                    try:
                        coeffs = polynomial_fit(x_data, y_data, deg)
                        poly = polynomial_function(coeffs)
                        r2 = r_squared(x_data, y_data, poly)
                        print(f"{'Polynomial (deg=' + str(deg) + ')':<25} {r2:.6f}")
                    except:
                        pass
            
            # Exponential
            try:
                a, b = exponential_fit(x_data, y_data)
                model = lambda x: a * math.exp(b * x)
                r2 = r_squared(x_data, y_data, model)
                print(f"{'Exponential':<25} {r2:.6f}")
            except:
                pass
            
            # Power
            try:
                a, b = power_fit(x_data, y_data)
                model = lambda x: a * (x ** b)
                r2 = r_squared(x_data, y_data, model)
                print(f"{'Power':<25} {r2:.6f}")
            except:
                pass
            
            # Logarithmic
            try:
                a, b = logarithmic_fit(x_data, y_data)
                model = lambda x: a + b * math.log(x)
                r2 = r_squared(x_data, y_data, model)
                print(f"{'Logarithmic':<25} {r2:.6f}")
            except:
                pass
                
        elif choice == '6':
            break
        else:
            print("Invalid choice!")


if __name__ == "__main__":
    main()