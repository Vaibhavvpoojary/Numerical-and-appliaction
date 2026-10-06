import numpy as np


def solve_bvp_finite_difference(p, q, r, a, b, alpha, beta, n):
    """
    Solve two-point linear BVP using finite difference method:
    y'' + p(x)y' + q(x)y = r(x),  a <= x <= b
    y(a) = alpha, y(b) = beta
    
    Parameters:
    p, q, r : functions of x (coefficients)
    a, b : domain endpoints
    alpha, beta : boundary conditions
    n : number of interior grid points
    
    Returns:
    x : grid points (including boundaries)
    y : solution at grid points
    """
    h = (b - a) / (n + 1)
    x = np.linspace(a, b, n + 2)
    
    # Construct tridiagonal matrix A and right-hand side b
    A = np.zeros((n, n))
    b_vec = np.zeros(n)
    
    for i in range(n):
        xi = x[i + 1]
        
        # Finite difference coefficients
        # y'' ≈ (y_{i-1} - 2y_i + y_{i+1}) / h^2
        # y' ≈ (y_{i+1} - y_{i-1}) / (2h)
        
        # Coefficient for y_{i-1}
        if i > 0:
            A[i, i-1] = 1/h**2 - p(xi)/(2*h)
        
        # Coefficient for y_i
        A[i, i] = -2/h**2 + q(xi)
        
        # Coefficient for y_{i+1}
        if i < n - 1:
            A[i, i+1] = 1/h**2 + p(xi)/(2*h)
        
        # Right-hand side
        b_vec[i] = r(xi)
    
    # Apply boundary conditions
    # For i=0, y_{i-1} = y_0 = alpha
    b_vec[0] -= (1/h**2 - p(x[1])/(2*h)) * alpha
    
    # For i=n-1, y_{i+1} = y_{n+1} = beta
    b_vec[-1] -= (1/h**2 + p(x[n])/(2*h)) * beta
    
    # Solve tridiagonal system
    y_interior = np.linalg.solve(A, b_vec)
    
    # Combine with boundary values
    y = np.concatenate(([alpha], y_interior, [beta]))
    
    return x, y


def thomas_algorithm(a, b, c, d):
    """
    Thomas algorithm for tridiagonal system.
    a: sub-diagonal (length n-1)
    b: diagonal (length n)
    c: super-diagonal (length n-1)
    d: right-hand side (length n)
    """
    n = len(b)
    c_prime = np.zeros(n-1)
    d_prime = np.zeros(n)
    
    # Forward sweep
    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]
    
    for i in range(1, n-1):
        denom = b[i] - a[i-1] * c_prime[i-1]
        c_prime[i] = c[i] / denom
        d_prime[i] = (d[i] - a[i-1] * d_prime[i-1]) / denom
    
    denom = b[n-1] - a[n-2] * c_prime[n-2]
    d_prime[n-1] = (d[n-1] - a[n-2] * d_prime[n-2]) / denom
    
    # Back substitution
    x = np.zeros(n)
    x[n-1] = d_prime[n-1]
    
    for i in range(n-2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i+1]
    
    return x


def solve_bvp_thomas(p, q, r, a, b, alpha, beta, n):
    """
    Solve BVP using Thomas algorithm (more efficient for tridiagonal).
    """
    h = (b - a) / (n + 1)
    x = np.linspace(a, b, n + 2)
    
    a_diag = np.zeros(n-1)  # sub-diagonal
    b_diag = np.zeros(n)    # diagonal
    c_diag = np.zeros(n-1)  # super-diagonal
    d_vec = np.zeros(n)     # RHS
    
    for i in range(n):
        xi = x[i + 1]
        
        if i > 0:
            a_diag[i-1] = 1/h**2 - p(xi)/(2*h)
        
        b_diag[i] = -2/h**2 + q(xi)
        
        if i < n - 1:
            c_diag[i] = 1/h**2 + p(xi)/(2*h)
        
        d_vec[i] = r(xi)
    
    # Boundary conditions
    d_vec[0] -= a_diag[0] * alpha
    d_vec[-1] -= c_diag[-1] * beta
    
    # Solve
    y_interior = thomas_algorithm(a_diag, b_diag, c_diag, d_vec)
    y = np.concatenate(([alpha], y_interior, [beta]))
    
    return x, y


def shooting_method(f, a, b, alpha, beta, guess1, guess2, tol=1e-6, max_iter=20):
    """
    Shooting method for nonlinear BVP: y'' = f(x, y, y')
    Convert to IVP and use secant method to find correct initial slope.
    
    Parameters:
    f : function f(x, y, yp) = y''
    a, b : domain endpoints
    alpha, beta : boundary conditions y(a)=alpha, y(b)=beta
    guess1, guess2 : two initial guesses for y'(a)
    tol : tolerance
    max_iter : maximum iterations
    
    Returns:
    x, y : solution
    """
    def solve_ivp(yp0):
        """Solve IVP using RK4."""
        n = 1000
        h = (b - a) / n
        x = np.linspace(a, b, n+1)
        y = np.zeros(n+1)
        yp = np.zeros(n+1)
        
        y[0] = alpha
        yp[0] = yp0
        
        for i in range(n):
            # System: y' = yp, yp' = f(x, y, yp)
            k1_y = h * yp[i]
            k1_yp = h * f(x[i], y[i], yp[i])
            
            k2_y = h * (yp[i] + k1_yp/2)
            k2_yp = h * f(x[i] + h/2, y[i] + k1_y/2, yp[i] + k1_yp/2)
            
            k3_y = h * (yp[i] + k2_yp/2)
            k3_yp = h * f(x[i] + h/2, y[i] + k2_y/2, yp[i] + k2_yp/2)
            
            k4_y = h * (yp[i] + k3_yp)
            k4_yp = h * f(x[i] + h, y[i] + k3_y, yp[i] + k3_yp)
            
            y[i+1] = y[i] + (k1_y + 2*k2_y + 2*k3_y + k4_y)/6
            yp[i+1] = yp[i] + (k1_yp + 2*k2_yp + 2*k3_yp + k4_yp)/6
        
        return x, y
    
    # Secant method
    yp_prev = guess1
    x, y = solve_ivp(yp_prev)
    f_prev = y[-1] - beta
    
    yp_curr = guess2
    x, y = solve_ivp(yp_curr)
    f_curr = y[-1] - beta
    
    for _ in range(max_iter):
        if abs(f_curr - f_prev) < 1e-12:
            break
        
        yp_new = yp_curr - f_curr * (yp_curr - yp_prev) / (f_curr - f_prev)
        
        x, y = solve_ivp(yp_new)
        f_new = y[-1] - beta
        
        if abs(f_new) < tol:
            return x, y
        
        yp_prev, f_prev = yp_curr, f_curr
        yp_curr, f_curr = yp_new, f_new
    
    return x, y


def main():
    print("=" * 60)
    print("Two-Point Boundary Value Problems - Finite Difference")
    print("=" * 60)
    
    examples = {
        '1': {
            'name': 'y\'\' = -y, y(0)=0, y(pi)=0  (Exact: y=0)',
            'p': lambda x: 0,
            'q': lambda x: 1,
            'r': lambda x: 0,
            'a': 0, 'b': np.pi,
            'alpha': 0, 'beta': 0,
            'exact': lambda x: 0*x
        },
        '2': {
            'name': 'y\'\' + y = x, y(0)=0, y(1)=0  (Exact: y = x - sin(x)/sin(1))',
            'p': lambda x: 0,
            'q': lambda x: 1,
            'r': lambda x: x,
            'a': 0, 'b': 1,
            'alpha': 0, 'beta': 0,
            'exact': lambda x: x - np.sin(x)/np.sin(1)
        },
        '3': {
            'name': 'y\'\' - 2y\' + y = x*e^x, y(0)=0, y(1)=e',
            'p': lambda x: -2,
            'q': lambda x: 1,
            'r': lambda x: x * np.exp(x),
            'a': 0, 'b': 1,
            'alpha': 0, 'beta': np.e,
            'exact': lambda x: x * np.exp(x)  # Particular solution
        },
        '4': {
            'name': 'Custom linear BVP',
            'p': None, 'q': None, 'r': None,
            'a': None, 'b': None,
            'alpha': None, 'beta': None,
            'exact': None
        }
    }
    
    while True:
        print("\nExamples:")
        for k, v in examples.items():
            print(f"  {k}. {v['name']}")
        print("  5. Shooting method (nonlinear)")
        print("  6. Exit")
        
        choice = input("\nSelect (1-6): ")
        
        if choice == '6':
            break
        
        if choice == '5':
            # Nonlinear shooting
            print("\nNonlinear BVP: y'' = f(x, y, y')")
            expr = input("Enter f(x, y, yp): ")
            f = lambda x, y, yp: eval(expr, {"np": np, "x": x, "y": y, "yp": yp})
            
            a = float(input("a: "))
            b = float(input("b: "))
            alpha = float(input("y(a): "))
            beta = float(input("y(b): "))
            guess1 = float(input("Guess 1 for y'(a): "))
            guess2 = float(input("Guess 2 for y'(a): "))
            
            x, y = shooting_method(f, a, b, alpha, beta, guess1, guess2)
            print(f"\nSolution found. y(b) = {y[-1]:.6f} (target: {beta})")
            continue
        
        if choice not in examples:
            print("Invalid!")
            continue
        
        ex = examples[choice]
        
        if choice == '4':
            ex['p'] = lambda x: eval(input("p(x) = "), {"np": np, "x": x})
            ex['q'] = lambda x: eval(input("q(x) = "), {"np": np, "x": x})
            ex['r'] = lambda x: eval(input("r(x) = "), {"np": np, "x": x})
            ex['a'] = float(input("a = "))
            ex['b'] = float(input("b = "))
            ex['alpha'] = float(input("y(a) = "))
            ex['beta'] = float(input("y(b) = "))
            exact_str = input("Exact solution (optional): ")
            if exact_str:
                ex['exact'] = lambda x: eval(exact_str, {"np": np, "x": x})
        
        n = int(input("\nNumber of interior points n: "))
        
        print(f"\n{'='*60}")
        print(f"Solving: {ex['name']}")
        print(f"{'='*60}")
        
        # Finite difference
        x_fd, y_fd = solve_bvp_finite_difference(ex['p'], ex['q'], ex['r'], 
                                                  ex['a'], ex['b'], ex['alpha'], ex['beta'], n)
        
        # Thomas algorithm
        x_th, y_th = solve_bvp_thomas(ex['p'], ex['q'], ex['r'], 
                                       ex['a'], ex['b'], ex['alpha'], ex['beta'], n)
        
        print(f"\n{'x':<10} {'FD':<15} {'Thomas':<15} {'Diff':<12}", end="")
        if ex['exact']:
            print(f" {'Exact':<15} {'Error FD':<12}")
        else:
            print()
        print("-" * 70)
        
        for i in range(len(x_fd)):
            diff = abs(y_fd[i] - y_th[i])
            if ex['exact']:
                exact = ex['exact'](x_fd[i])
                err = abs(y_fd[i] - exact)
                print(f"{x_fd[i]:<10.4f} {y_fd[i]:<15.8f} {y_th[i]:<15.8f} {diff:<12.2e} {exact:<15.8f} {err:<12.2e}")
            else:
                print(f"{x_fd[i]:<10.4f} {y_fd[i]:<15.8f} {y_th[i]:<15.8f} {diff:<12.2e}")


if __name__ == "__main__":
    main()