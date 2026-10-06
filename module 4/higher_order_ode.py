import math


def rk4_system(f_system, x0, y0, h, n):
    """
    4th-order Runge-Kutta for system of first-order ODEs.
    
    Parameters:
    f_system : list of functions [f1, f2, ..., fm] where
               fi(x, y1, y2, ..., ym) = dyi/dx
    x0 : initial x
    y0 : list of initial values [y1_0, y2_0, ..., ym_0]
    h : step size
    n : number of steps
    
    Returns:
    x_vals, y_vals where y_vals is list of lists [[y1_0, y2_0, ...], [y1_1, y2_1, ...], ...]
    """
    m = len(y0)
    x = x0
    y = y0[:]
    x_vals = [x]
    y_vals = [y[:]]
    
    for i in range(n):
        # k1
        k1 = [h * f(x, *y) for f in f_system]
        
        # k2
        y_mid = [y[j] + k1[j]/2 for j in range(m)]
        k2 = [h * f(x + h/2, *y_mid) for f in f_system]
        
        # k3
        y_mid = [y[j] + k2[j]/2 for j in range(m)]
        k3 = [h * f(x + h/2, *y_mid) for f in f_system]
        
        # k4
        y_end = [y[j] + k3[j] for j in range(m)]
        k4 = [h * f(x + h, *y_end) for f in f_system]
        
        # Update
        y = [y[j] + (k1[j] + 2*k2[j] + 2*k3[j] + k4[j])/6 for j in range(m)]
        x += h
        x_vals.append(x)
        y_vals.append(y[:])
    
    return x_vals, y_vals


def higher_order_to_system(f, order):
    """
    Convert nth-order ODE y^(n) = f(x, y, y', ..., y^(n-1)) to system of first-order ODEs.
    
    Returns:
    f_system : list of functions for the system
    """
    # y1 = y, y2 = y', y3 = y'', ..., yn = y^(n-1)
    # y1' = y2
    # y2' = y3
    # ...
    # yn-1' = yn
    # yn' = f(x, y1, y2, ..., yn)
    
    f_system = []
    for i in range(order - 1):
        f_system.append(lambda x, *y, idx=i: y[idx + 1])
    f_system.append(f)
    
    return f_system


def solve_higher_order_rk4(f, x0, initial_conditions, h, n):
    """
    Solve nth-order ODE using RK4.
    
    Parameters:
    f : function f(x, y, y', y'', ...) = y^(n)
    x0 : initial x
    initial_conditions : list [y0, y0', y0'', ..., y0^(n-1)]
    h : step size
    n : number of steps
    
    Returns:
    x_vals, y_vals (y_vals[i][0] is the solution y)
    """
    order = len(initial_conditions)
    f_system = higher_order_to_system(f, order)
    return rk4_system(f_system, x0, initial_conditions, h, n)


def euler_system(f_system, x0, y0, h, n):
    """Euler's method for system of ODEs."""
    m = len(y0)
    x = x0
    y = y0[:]
    x_vals = [x]
    y_vals = [y[:]]
    
    for i in range(n):
        dy = [f(x, *y) for f in f_system]
        y = [y[j] + h * dy[j] for j in range(m)]
        x += h
        x_vals.append(x)
        y_vals.append(y[:])
    
    return x_vals, y_vals


# Predefined examples
EXAMPLES = {
    '1': {
        'name': 'Second-order: y\'\' + y = 0  (Simple Harmonic Motion)',
        'desc': 'y\'\' = -y,  y(0)=1, y\'(0)=0  =>  y = cos(x)',
        'order': 2,
        'f': lambda x, y, yp: -y,
        'initial': [1, 0],
        'exact_y': lambda x: math.cos(x),
        'exact_yp': lambda x: -math.sin(x)
    },
    '2': {
        'name': 'Second-order: y\'\' - y = 0',
        'desc': 'y\'\' = y,  y(0)=1, y\'(0)=0  =>  y = cosh(x)',
        'order': 2,
        'f': lambda x, y, yp: y,
        'initial': [1, 0],
        'exact_y': lambda x: math.cosh(x),
        'exact_yp': lambda x: math.sinh(x)
    },
    '3': {
        'name': 'Third-order: y\'\'\' = -y\'',
        'desc': 'y\'\'\' = -y\',  y(0)=1, y\'(0)=0, y\'\'(0)=-1  =>  y = cos(x)',
        'order': 3,
        'f': lambda x, y, yp, ypp: -yp,
        'initial': [1, 0, -1],
        'exact_y': lambda x: math.cos(x),
        'exact_yp': lambda x: -math.sin(x)
    },
    '4': {
        'name': 'System: Predator-Prey (Lotka-Volterra)',
        'desc': 'dx/dt = ax - bxy,  dy/dt = cxy - dy',
        'order': 'system',
        'f_system': [
            lambda t, x, y: 1.5*x - 1*x*y,
            lambda t, x, y: 1*x*y - 3*y
        ],
        'initial': [10, 5],
        't_span': (0, 15)
    },
    '5': {
        'name': 'System: Damped Harmonic Oscillator',
        'desc': 'y\'\' + 2*beta*y\' + omega^2*y = 0  ->  y1\'=y2, y2\'=-2*beta*y2 - omega^2*y1',
        'order': 2,
        'f': lambda x, y, yp: -2*0.1*yp - 1*y,
        'initial': [1, 0],
        'exact_y': lambda x: math.exp(-0.1*x) * math.cos(math.sqrt(0.99)*x)
    },
    '6': {
        'name': 'Custom ODE/System',
        'f': None,
        'f_system': None,
        'initial': None,
        'order': None
    }
}


def print_system_results(name, x_vals, y_vals, exact_y=None, exact_components=None):
    """Print results for system/higher-order ODE."""
    print(f"\n{name}:")
    print("-" * 80)
    m = len(y_vals[0])
    
    if m == 2 and exact_y and exact_components:
        print(f"{'x':<8} {'y':<14} {'y_exact':<14} {'Error':<12} {'y\'':<14} {'y\'_exact':<14} {'Error':<12}")
        for x, y in zip(x_vals, y_vals):
            y_ex = exact_y(x)
            yp_ex = exact_components(x)
            print(f"{x:<8.4f} {y[0]:<14.8f} {y_ex:<14.8f} {abs(y[0]-y_ex):<12.2e} "
                  f"{y[1]:<14.8f} {yp_ex:<14.8f} {abs(y[1]-yp_ex):<12.2e}")
    elif exact_y:
        print(f"{'x':<8} {'y':<14} {'y_exact':<14} {'Error':<12}")
        for x, y in zip(x_vals, y_vals):
            y_ex = exact_y(x)
            print(f"{x:<8.4f} {y[0]:<14.8f} {y_ex:<14.8f} {abs(y[0]-y_ex):<12.2e}")
    else:
        headers = ['x'] + [f'y{i+1}' for i in range(m)]
        print('  '.join(f'{h:<14}' for h in headers))
        for x, y in zip(x_vals, y_vals):
            row = [f'{x:<8.4f}'] + [f'{yi:<14.8f}' for yi in y]
            print('  '.join(row))
    print("-" * 80)


def main():
    print("=" * 60)
    print("Higher-Order ODEs and Systems - RK4 Method")
    print("=" * 60)
    
    while True:
        print("\nExamples:")
        for k, v in EXAMPLES.items():
            print(f"  {k}. {v['name']}")
            print(f"     {v['desc']}")
        print("  7. Exit")
        
        choice = input("\nSelect example (1-7): ")
        
        if choice == '7':
            break
        
        if choice not in EXAMPLES:
            print("Invalid choice!")
            continue
        
        ex = EXAMPLES[choice]
        
        if choice == '6':
            print("Custom system/higher-order ODE:")
            sys_type = input("Type (system/higher): ").lower()
            
            if sys_type == 'system':
                m = int(input("Number of equations: "))
                ex['f_system'] = []
                for i in range(m):
                    expr = input(f"  dy{i+1}/dt = ")
                    ex['f_system'].append(lambda t, *y, e=expr: eval(e, {"math": math, "t": t, **{f'y{j+1}': y[j] for j in range(m)}}))
                ex['initial'] = list(map(float, input("Initial values (comma-separated): ").split(',')))
                ex['order'] = 'system'
            else:
                order = int(input("Order of ODE: "))
                expr = input(f"  y^({order}) = ")
                ex['f'] = lambda x, *y, e=expr: eval(e, {"math": math, "x": x, **{f'y{j}': y[j] for j in range(order)}})
                ex['initial'] = list(map(float, input("Initial conditions [y, y', ...]: ").split(',')))
                ex['order'] = order
        
        h = float(input("\nStep size h: "))
        n = int(input("Number of steps: "))
        
        print(f"\n{'='*60}")
        print(f"Solving: {ex['name']}")
        print(f"Initial: {ex['initial']}")
        print(f"h = {h}, n = {n}")
        print(f"{'='*60}")
        
        if ex.get('order') == 'system':
            f_system = ex['f_system']
            x_vals, y_vals = rk4_system(f_system, 0, ex['initial'], h, n)
            
            if choice == '4':  # Lotka-Volterra
                print_system_results("RK4 - Lotka-Volterra", x_vals, y_vals)
                # Also run Euler for comparison
                x_e, y_e = euler_system(f_system, 0, ex['initial'], h, n)
                print_system_results("Euler - Lotka-Volterra", x_e, y_e)
            else:
                print_system_results("RK4", x_vals, y_vals)
                
        else:
            order = ex['order']
            f = ex['f']
            initial = ex['initial']
            exact_y = ex.get('exact_y')
            exact_yp = ex.get('exact_components') or ex.get('exact_yp')
            
            x_vals, y_vals = solve_higher_order_rk4(f, 0, initial, h, n)
            
            print_system_results("RK4", x_vals, y_vals, exact_y, exact_yp)
            
            # Compare with Euler
            f_system = higher_order_to_system(f, order)
            x_e, y_e = euler_system(f_system, 0, initial, h, n)
            print_system_results("Euler", x_e, y_e, exact_y, exact_yp)


if __name__ == "__main__":
    main()