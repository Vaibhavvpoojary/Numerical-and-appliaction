import math


# Gauss-Legendre quadrature nodes and weights for standard interval [-1, 1]
GAUSS_LEGENDRE = {
    2: {
        'nodes': [-1/math.sqrt(3), 1/math.sqrt(3)],
        'weights': [1.0, 1.0]
    },
    3: {
        'nodes': [-math.sqrt(3/5), 0.0, math.sqrt(3/5)],
        'weights': [5/9, 8/9, 5/9]
    },
    4: {
        'nodes': [
            -math.sqrt(3/7 + 2/7*math.sqrt(6/5)),
            -math.sqrt(3/7 - 2/7*math.sqrt(6/5)),
            math.sqrt(3/7 - 2/7*math.sqrt(6/5)),
            math.sqrt(3/7 + 2/7*math.sqrt(6/5))
        ],
        'weights': [
            (18 - math.sqrt(30))/36,
            (18 + math.sqrt(30))/36,
            (18 + math.sqrt(30))/36,
            (18 - math.sqrt(30))/36
        ]
    },
    5: {
        'nodes': [
            -1/3 * math.sqrt(5 + 2*math.sqrt(10/7)),
            -1/3 * math.sqrt(5 - 2*math.sqrt(10/7)),
            0.0,
            1/3 * math.sqrt(5 - 2*math.sqrt(10/7)),
            1/3 * math.sqrt(5 + 2*math.sqrt(10/7))
        ],
        'weights': [
            (322 - 13*math.sqrt(70))/900,
            (322 + 13*math.sqrt(70))/900,
            128/225,
            (322 + 13*math.sqrt(70))/900,
            (322 - 13*math.sqrt(70))/900
        ]
    }
}


def gauss_quadrature(f, a, b, n=3):
    """
    Gaussian quadrature for integral on [a, b].
    
    Parameters:
    f : function to integrate
    a : lower limit
    b : upper limit
    n : number of points (2, 3, 4, or 5)
    
    Returns:
    Approximate integral value
    """
    if n not in GAUSS_LEGENDRE:
        raise ValueError(f"n={n} not supported. Use 2, 3, 4, or 5.")
    
    nodes = GAUSS_LEGENDRE[n]['nodes']
    weights = GAUSS_LEGENDRE[n]['weights']
    
    # Transform from [-1, 1] to [a, b]
    # x = (b-a)/2 * t + (a+b)/2
    # dx = (b-a)/2 * dt
    scale = (b - a) / 2
    shift = (a + b) / 2
    
    integral = 0.0
    for t, w in zip(nodes, weights):
        x = scale * t + shift
        integral += w * f(x)
    
    return integral * scale


def gauss_quadrature_2pt(f, a, b):
    """2-point Gaussian quadrature (exact for polynomials up to degree 3)."""
    return gauss_quadrature(f, a, b, 2)


def gauss_quadrature_3pt(f, a, b):
    """3-point Gaussian quadrature (exact for polynomials up to degree 5)."""
    return gauss_quadrature(f, a, b, 3)


def gauss_quadrature_double(f, ax, bx, ay, by, nx=3, ny=3):
    """
    Gaussian quadrature for double integral.
    
    Parameters:
    f : function of two variables f(x, y)
    ax, bx : x limits
    ay, by : y limits
    nx, ny : number of Gauss points in x and y directions
    
    Returns:
    Approximate double integral value
    """
    if nx not in GAUSS_LEGENDRE or ny not in GAUSS_LEGENDRE:
        raise ValueError("nx and ny must be 2, 3, 4, or 5")
    
    nodes_x = GAUSS_LEGENDRE[nx]['nodes']
    weights_x = GAUSS_LEGENDRE[nx]['weights']
    nodes_y = GAUSS_LEGENDRE[ny]['nodes']
    weights_y = GAUSS_LEGENDRE[ny]['weights']
    
    # Transform from [-1, 1] to [ax, bx] and [ay, by]
    scale_x = (bx - ax) / 2
    shift_x = (ax + bx) / 2
    scale_y = (by - ay) / 2
    shift_y = (ay + by) / 2
    
    integral = 0.0
    for tx, wx in zip(nodes_x, weights_x):
        x = scale_x * tx + shift_x
        for ty, wy in zip(nodes_y, weights_y):
            y = scale_y * ty + shift_y
            integral += wx * wy * f(x, y)
    
    return integral * scale_x * scale_y


def gauss_legendre_nodes_weights(n):
    """
    Return nodes and weights for n-point Gauss-Legendre quadrature.
    """
    if n not in GAUSS_LEGENDRE:
        raise ValueError(f"n={n} not supported")
    return GAUSS_LEGENDRE[n]['nodes'], GAUSS_LEGENDRE[n]['weights']


def legendre_poly(n, x):
    """Legendre polynomial P_n(x) using recurrence relation."""
    if n == 0:
        return 1.0
    if n == 1:
        return x
    
    P0 = 1.0
    P1 = x
    for k in range(2, n + 1):
        Pk = ((2*k - 1) * x * P1 - (k - 1) * P0) / k
        P0, P1 = P1, Pk
    return P1


def legendre_derivative(n, x):
    """Derivative of Legendre polynomial P_n'(x)."""
    if n == 0:
        return 0.0
    return n / (x*x - 1) * (x * legendre_poly(n, x) - legendre_poly(n-1, x))


def compute_gauss_nodes_weights(n, tol=1e-15, max_iter=100):
    """
    Compute Gauss-Legendre nodes and weights for n points using Newton-Raphson.
    Nodes are roots of Legendre polynomial P_n(x).
    Weights are 2 / [(1-x^2) * (P_n'(x))^2]
    """
    nodes = []
    weights = []
    
    # Initial guesses for roots of P_n(x) using Chebyshev nodes
    for i in range(1, n + 1):
        x = math.cos(math.pi * (i - 0.25) / (n + 0.5))
        
        # Newton-Raphson to find root
        for _ in range(max_iter):
            Pn = legendre_poly(n, x)
            Pn_prime = legendre_derivative(n, x)
            dx = -Pn / Pn_prime
            x += dx
            if abs(dx) < tol:
                break
        
        nodes.append(x)
        w = 2 / ((1 - x*x) * Pn_prime * Pn_prime)
        weights.append(w)
    
    # Sort nodes and corresponding weights
    pairs = sorted(zip(nodes, weights))
    nodes = [p[0] for p in pairs]
    weights = [p[1] for p in pairs]
    
    return nodes, weights


def main():
    print("=" * 60)
    print("Gaussian Quadrature (Gauss-Legendre)")
    print("=" * 60)
    
    print("\nPre-computed nodes and weights:")
    for n in [2, 3, 4, 5]:
        nodes, weights = gauss_legendre_nodes_weights(n)
        print(f"\n{n}-point:")
        for i, (node, weight) in enumerate(zip(nodes, weights)):
            print(f"  x_{i+1} = {node:.10f},  w_{i+1} = {weight:.10f}")
    
    while True:
        print("\nOptions:")
        print("1. Single integral (2-point)")
        print("2. Single integral (3-point)")
        print("3. Single integral (n-point, n=2-5)")
        print("4. Double integral")
        print("5. Compare methods")
        print("6. Exit")
        choice = input("Enter choice (1-6): ")
        
        if choice in ['1', '2', '3']:
            expr = input("Enter function f(x) (e.g., 'x**2', 'math.sin(x)', 'math.exp(-x**2)'): ")
            f = lambda x: eval(expr, {"math": math, "x": x})
            
            a = float(input("Enter lower limit a: "))
            b = float(input("Enter upper limit b: "))
            
            if choice == '1':
                result = gauss_quadrature_2pt(f, a, b)
                print(f"\n2-point Gauss: {result:.12f}")
            elif choice == '2':
                result = gauss_quadrature_3pt(f, a, b)
                print(f"\n3-point Gauss: {result:.12f}")
            else:
                n = int(input("Enter n (2, 3, 4, or 5): "))
                try:
                    result = gauss_quadrature(f, a, b, n)
                    print(f"\n{n}-point Gauss: {result:.12f}")
                except ValueError as e:
                    print(f"Error: {e}")
                    
        elif choice == '4':
            expr = input("Enter function f(x,y): ")
            f = lambda x, y: eval(expr, {"math": math, "x": x, "y": y})
            
            ax = float(input("Enter x lower limit: "))
            bx = float(input("Enter x upper limit: "))
            ay = float(input("Enter y lower limit: "))
            by = float(input("Enter y upper limit: "))
            nx = int(input("Enter nx (2-5): "))
            ny = int(input("Enter ny (2-5): "))
            
            try:
                result = gauss_quadrature_double(f, ax, bx, ay, by, nx, ny)
                print(f"\nDouble integral ({nx}x{ny} points): {result:.12f}")
            except ValueError as e:
                print(f"Error: {e}")
                
        elif choice == '5':
            expr = input("Enter function f(x): ")
            f = lambda x: eval(expr, {"math": math, "x": x})
            
            a = float(input("Enter lower limit a: "))
            b = float(input("Enter upper limit b: "))
            
            # Exact integral for comparison (if known)
            exact = input("Enter exact value (or press Enter to skip): ")
            exact_val = float(exact) if exact else None
            
            print(f"\n{'Method':<20} {'Result':<20} {'Error':<15}")
            print("-" * 55)
            
            for n in [2, 3, 4, 5]:
                try:
                    result = gauss_quadrature(f, a, b, n)
                    error = abs(result - exact_val) if exact_val else 'N/A'
                    print(f"{n}-point Gauss        {result:.12f}  {error if isinstance(error, str) else f'{error:.2e}'}")
                except:
                    pass
            
        elif choice == '6':
            break
        else:
            print("Invalid choice!")


if __name__ == "__main__":
    main()