import math


def taylor_series_method(f, df, x0, y0, h, n, order=4):
    """
    Taylor series method for solving dy/dx = f(x, y).
    
    Parameters:
    f : function f(x, y) = dy/dx
    df : list of derivative functions [df1, df2, ...] where
         df1 = f, df2 = d^2y/dx^2, df3 = d^3y/dx^3, etc.
    x0 : initial x
    y0 : initial y
    h : step size
    n : number of steps
    order : order of Taylor series (default 4)
    
    Returns:
    (x_values, y_values) lists
    """
    x = x0
    y = y0
    x_vals = [x]
    y_vals = [y]
    
    for i in range(n):
        # Compute Taylor series terms
        y_new = y
        term = 0.0
        
        for k in range(1, order + 1):
            if k - 1 < len(df):
                term = df[k-1](x, y)
            else:
                break
            term *= h**k / math.factorial(k)
            y_new += term
        
        y = y_new
        x += h
        x_vals.append(x)
        y_vals.append(y)
    
    return x_vals, y_vals


def euler_method(f, x0, y0, h, n):
    """
    Euler's method (forward Euler) for solving dy/dx = f(x, y).
    
    Parameters:
    f : function f(x, y) = dy/dx
    x0 : initial x
    y0 : initial y
    h : step size
    n : number of steps
    
    Returns:
    (x_values, y_values) lists
    """
    x = x0
    y = y0
    x_vals = [x]
    y_vals = [y]
    
    for i in range(n):
        y = y + h * f(x, y)
        x = x + h
        x_vals.append(x)
        y_vals.append(y)
    
    return x_vals, y_vals


def modified_euler_method(f, x0, y0, h, n):
    """
    Modified Euler's method (Heun's method) for solving dy/dx = f(x, y).
    
    Parameters:
    f : function f(x, y) = dy/dx
    x0 : initial x
    y0 : initial y
    h : step size
    n : number of steps
    
    Returns:
    (x_values, y_values) lists
    """
    x = x0
    y = y0
    x_vals = [x]
    y_vals = [y]
    
    for i in range(n):
        # Predictor step
        y_pred = y + h * f(x, y)
        # Corrector step
        y = y + (h / 2) * (f(x, y) + f(x + h, y_pred))
        x = x + h
        x_vals.append(x)
        y_vals.append(y)
    
    return x_vals, y_vals


def runge_kutta_4(f, x0, y0, h, n):
    """
    Classical 4th-order Runge-Kutta method for solving dy/dx = f(x, y).
    
    Parameters:
    f : function f(x, y) = dy/dx
    x0 : initial x
    y0 : initial y
    h : step size
    n : number of steps
    
    Returns:
    (x_values, y_values) lists
    """
    x = x0
    y = y0
    x_vals = [x]
    y_vals = [y]
    
    for i in range(n):
        k1 = h * f(x, y)
        k2 = h * f(x + h/2, y + k1/2)
        k3 = h * f(x + h/2, y + k2/2)
        k4 = h * f(x + h, y + k3)
        
        y = y + (k1 + 2*k2 + 2*k3 + k4) / 6
        x = x + h
        x_vals.append(x)
        y_vals.append(y)
    
    return x_vals, y_vals


def runge_kutta_4_system(f_system, x0, y0, h, n):
    """
    4th-order Runge-Kutta for system of first-order ODEs.
    
    Parameters:
    f_system : list of functions [f1, f2, ...] where
               f1(x, y1, y2, ...) = dy1/dx, etc.
    x0 : initial x
    y0 : list of initial y values
    h : step size
    n : number of steps
    
    Returns:
    (x_values, y_values) where y_values is list of lists
    """
    m = len(y0)
    x = x0
    y = y0[:]
    x_vals = [x]
    y_vals = [y[:]]
    
    for i in range(n):
        k1 = [h * f(x, *y) for f in f_system]
        
        y_mid = [y[j] + k1[j]/2 for j in range(m)]
        k2 = [h * f(x + h/2, *y_mid) for f in f_system]
        
        y_mid = [y[j] + k2[j]/2 for j in range(m)]
        k3 = [h * f(x + h/2, *y_mid) for f in f_system]
        
        y_end = [y[j] + k3[j] for j in range(m)]
        k4 = [h * f(x + h, *y_end) for f in f_system]
        
        y = [y[j] + (k1[j] + 2*k2[j] + 2*k3[j] + k4[j])/6 for j in range(m)]
        x = x + h
        x_vals.append(x)
        y_vals.append(y[:])
    
    return x_vals, y_vals


def solve_higher_order_rk4(f, x0, y0, y0_prime, h, n):
    """
    Solve second-order ODE y'' = f(x, y, y') using RK4 by converting to system.
    
    Parameters:
    f : function f(x, y, y') = y''
    x0 : initial x
    y0 : initial y
    y0_prime : initial y'
    h : step size
    n : number of steps
    
    Returns:
    (x_values, y_values, y_prime_values)
    """
    # Convert to system: y1 = y, y2 = y'
    # y1' = y2
    # y2' = f(x, y1, y2)
    f1 = lambda x, y1, y2: y2
    f2 = lambda x, y1, y2: f(x, y1, y2)
    
    x_vals, y_system = runge_kutta_4_system([f1, f2], x0, [y0, y0_prime], h, n)
    
    y_vals = [y[0] for y in y_system]
    y_prime_vals = [y[1] for y in y_system]
    
    return x_vals, y_vals, y_prime_vals


def adaptive_rk4(f, x0, y0, x_end, h_init, tol=1e-6):
    """
    Adaptive step-size RK4 with error control.
    
    Parameters:
    f : function f(x, y) = dy/dx
    x0 : initial x
    y0 : initial y
    x_end : final x
    h_init : initial step size
    tol : error tolerance
    
    Returns:
    (x_values, y_values)
    """
    x = x0
    y = y0
    h = h_init
    x_vals = [x]
    y_vals = [y]
    
    while x < x_end:
        if x + h > x_end:
            h = x_end - x
        
        # Single step of size h
        k1 = h * f(x, y)
        k2 = h * f(x + h/2, y + k1/2)
        k3 = h * f(x + h/2, y + k2/2)
        k4 = h * f(x + h, y + k3)
        y1 = y + (k1 + 2*k2 + 2*k3 + k4) / 6
        
        # Two steps of size h/2
        h2 = h / 2
        k1 = h2 * f(x, y)
        k2 = h2 * f(x + h2/2, y + k1/2)
        k3 = h2 * f(x + h2/2, y + k2/2)
        k4 = h2 * f(x + h2, y + k3)
        y_mid = y + (k1 + 2*k2 + 2*k3 + k4) / 6
        
        k1 = h2 * f(x + h2, y_mid)
        k2 = h2 * f(x + 3*h2/2, y_mid + k1/2)
        k3 = h2 * f(x + 3*h2/2, y_mid + k2/2)
        k4 = h2 * f(x + 2*h2, y_mid + k3)
        y2 = y_mid + (k1 + 2*k2 + 2*k3 + k4) / 6
        
        # Error estimate
        error = abs(y2 - y1)
        
        if error < tol:
            # Accept step
            x += h
            y = y2 + (y2 - y1) / 15  # Richardson extrapolation
            x_vals.append(x)
            y_vals.append(y)
            
            # Adjust step size
            if error > 0:
                h *= min(2.0, max(0.5, 0.9 * (tol / error)**0.2))
        else:
            # Reject step, reduce h
            h *= max(0.5, 0.9 * (tol / error)**0.2)
    
    return x_vals, y_vals


def print_results(method_name, x_vals, y_vals, exact_func=None):
    """Print results in a formatted table."""
    print(f"\n{method_name} Results:")
    print("-" * 60)
    if exact_func:
        print(f"{'x':<10} {'y (approx)':<15} {'y (exact)':<15} {'Error':<15}")
    else:
        print(f"{'x':<10} {'y (approx)':<15}")
    print("-" * 60)
    
    for x, y in zip(x_vals, y_vals):
        if exact_func:
            y_exact = exact_func(x)
            error = abs(y - y_exact)
            print(f"{x:<10.4f} {y:<15.8f} {y_exact:<15.8f} {error:<15.2e}")
        else:
            print(f"{x:<10.4f} {y:<15.8f}")
    print("-" * 60)


def main():
    print("=" * 60)
    print("ODE Solvers: Taylor Series, Euler, Modified Euler, RK4")
    print("=" * 60)
    
    # Example ODEs
    examples = {
        '1': {
            'name': "dy/dx = x + y, y(0) = 1",
            'f': lambda x, y: x + y,
            'exact': lambda x: 2*math.exp(x) - x - 1,
            'df': [
                lambda x, y: x + y,
                lambda x, y: 1 + x + y,
                lambda x, y: 1 + x + y,
                lambda x, y: 1 + x + y
            ]
        },
        '2': {
            'name': "dy/dx = y, y(0) = 1 (exponential)",
            'f': lambda x, y: y,
            'exact': lambda x: math.exp(x),
            'df': [
                lambda x, y: y,
                lambda x, y: y,
                lambda x, y: y,
                lambda x, y: y
            ]
        },
        '3': {
            'name': "dy/dx = -2xy, y(0) = 1 (Gaussian)",
            'f': lambda x, y: -2*x*y,
            'exact': lambda x: math.exp(-x*x),
            'df': [
                lambda x, y: -2*x*y,
                lambda x, y: -2*y + 4*x*x*y,
                lambda x, y: -12*x*y + 8*x*x*x*y,
                lambda x, y: -12*y + 48*x*x*y - 16*x*x*x*x*y
            ]
        },
        '4': {
            'name': "Custom ODE",
            'f': None,
            'exact': None,
            'df': None
        }
    }
    
    while True:
        print("\nExample ODEs:")
        for k, v in examples.items():
            print(f"  {k}. {v['name']}")
        print("  5. Exit")
        
        choice = input("\nSelect ODE (1-5): ")
        
        if choice == '5':
            break
        
        if choice not in examples:
            print("Invalid choice!")
            continue
        
        ex = examples[choice]
        
        if choice == '4':
            expr = input("Enter f(x,y) (e.g., 'x + y', 'x*y', '-y'): ")
            ex['f'] = lambda x, y: eval(expr, {"math": math, "x": x, "y": y})
            exact_input = input("Enter exact solution (or press Enter to skip): ")
            if exact_input:
                ex['exact'] = lambda x: eval(exact_input, {"math": math, "x": x})
            print("Taylor series requires derivatives. Enter up to 4 derivatives:")
            ex['df'] = []
            for i in range(4):
                deriv = input(f"  d^{i+1}y/dx^{i+1} (or Enter to stop): ")
                if not deriv:
                    break
                ex['df'].append(lambda x, y, d=deriv: eval(d, {"math": math, "x": x, "y": y}))
        
        f = ex['f']
        exact = ex['exact']
        df = ex['df']
        
        x0 = float(input("\nEnter initial x0: "))
        y0 = float(input("Enter initial y0: "))
        h = float(input("Enter step size h: "))
        n = int(input("Enter number of steps n: "))
        
        print(f"\n{'='*60}")
        print(f"Solving: {ex['name']}")
        print(f"Initial condition: y({x0}) = {y0}")
        print(f"Step size: h = {h}, Steps: n = {n}")
        print(f"{'='*60}")
        
        # Taylor Series
        if df:
            order = min(4, len(df))
            x_t, y_t = taylor_series_method(f, df, x0, y0, h, n, order)
            print_results(f"Taylor Series (order {order})", x_t, y_t, exact)
        
        # Euler
        x_e, y_e = euler_method(f, x0, y0, h, n)
        print_results("Euler's Method", x_e, y_e, exact)
        
        # Modified Euler
        x_me, y_me = modified_euler_method(f, x0, y0, h, n)
        print_results("Modified Euler (Heun)", x_me, y_me, exact)
        
        # RK4
        x_rk, y_rk = runge_kutta_4(f, x0, y0, h, n)
        print_results("Runge-Kutta 4th Order", x_rk, y_rk, exact)
        
        # Compare at final point
        print("\nFinal value comparison at x = {:.4f}:".format(x0 + n*h))
        if exact:
            exact_final = exact(x0 + n*h)
            print(f"  Exact:              {exact_final:.10f}")
        if df:
            print(f"  Taylor (order {order}): {y_t[-1]:.10f}  Error: {abs(y_t[-1] - exact_final):.2e}" if exact else f"  Taylor: {y_t[-1]:.10f}")
        print(f"  Euler:              {y_e[-1]:.10f}  Error: {abs(y_e[-1] - exact_final):.2e}" if exact else f"  Euler: {y_e[-1]:.10f}")
        print(f"  Mod. Euler:         {y_me[-1]:.10f}  Error: {abs(y_me[-1] - exact_final):.2e}" if exact else f"  Mod. Euler: {y_me[-1]:.10f}")
        print(f"  RK4:                {y_rk[-1]:.10f}  Error: {abs(y_rk[-1] - exact_final):.2e}" if exact else f"  RK4: {y_rk[-1]:.10f}")


if __name__ == "__main__":
    main()