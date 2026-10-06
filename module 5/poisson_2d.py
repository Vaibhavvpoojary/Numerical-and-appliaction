import numpy as np


def laplace_2d_gs(nx, ny, Lx, Ly, bc_left, bc_right, bc_bottom, bc_top, 
                   max_iter=10000, tol=1e-6, omega=1.0):
    """
    Solve 2D Laplace equation ∇²u = 0 using Gauss-Seidel (or SOR if omega != 1).
    
    Parameters:
    nx, ny : grid points in x and y
    Lx, Ly : domain size
    bc_left, bc_right : boundary condition functions of y (for x=0, x=Lx)
    bc_bottom, bc_top : boundary condition functions of x (for y=0, y=Ly)
    max_iter : maximum iterations
    tol : convergence tolerance
    omega : relaxation parameter (1.0 = Gauss-Seidel, >1.0 = SOR)
    
    Returns:
    x, y, u : grid and solution
    """
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    u = np.zeros((ny, nx))
    
    # Apply boundary conditions
    u[:, 0] = bc_left(y)
    u[:, -1] = bc_right(y)
    u[0, :] = bc_bottom(x)
    u[-1, :] = bc_top(x)
    
    # Gauss-Seidel / SOR iteration
    for iteration in range(max_iter):
        max_diff = 0.0
        
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                # Standard 5-point stencil
                u_new = (dy**2 * (u[j, i+1] + u[j, i-1]) + 
                         dx**2 * (u[j+1, i] + u[j-1, i])) / (2 * (dx**2 + dy**2))
                
                # SOR update
                u_new = u[j, i] + omega * (u_new - u[j, i])
                
                diff = abs(u_new - u[j, i])
                if diff > max_diff:
                    max_diff = diff
                
                u[j, i] = u_new
        
        if max_diff < tol:
            print(f"Converged after {iteration + 1} iterations, max_diff = {max_diff:.2e}")
            break
    else:
        print(f"Max iterations reached, max_diff = {max_diff:.2e}")
    
    return x, y, u


def laplace_2d_jacobi(nx, ny, Lx, Ly, bc_left, bc_right, bc_bottom, bc_top,
                       max_iter=10000, tol=1e-6):
    """
    Solve 2D Laplace equation using Jacobi method.
    """
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    u = np.zeros((ny, nx))
    u_new = np.zeros((ny, nx))
    
    # Apply boundary conditions
    u[:, 0] = bc_left(y)
    u[:, -1] = bc_right(y)
    u[0, :] = bc_bottom(x)
    u[-1, :] = bc_top(x)
    u_new[:, 0] = u[:, 0]
    u_new[:, -1] = u[:, -1]
    u_new[0, :] = u[0, :]
    u_new[-1, :] = u[-1, :]
    
    for iteration in range(max_iter):
        max_diff = 0.0
        
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                u_new[j, i] = (dy**2 * (u[j, i+1] + u[j, i-1]) + 
                               dx**2 * (u[j+1, i] + u[j-1, i])) / (2 * (dx**2 + dy**2))
                
                diff = abs(u_new[j, i] - u[j, i])
                if diff > max_diff:
                    max_diff = diff
        
        u[1:-1, 1:-1] = u_new[1:-1, 1:-1]
        
        if max_diff < tol:
            print(f"Jacobi converged after {iteration + 1} iterations")
            break
    
    return x, y, u


def poisson_2d_gs(nx, ny, Lx, Ly, f, bc_left, bc_right, bc_bottom, bc_top,
                   max_iter=10000, tol=1e-6, omega=1.0):
    """
    Solve 2D Poisson equation ∇²u = f(x,y) using Gauss-Seidel/SOR.
    
    Parameters:
    f : source function f(x, y)
    Other params same as laplace_2d_gs
    """
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    u = np.zeros((ny, nx))
    
    # Apply boundary conditions
    u[:, 0] = bc_left(y)
    u[:, -1] = bc_right(y)
    u[0, :] = bc_bottom(x)
    u[-1, :] = bc_top(x)
    
    # Precompute source term
    f_vals = np.zeros((ny, nx))
    for j in range(ny):
        for i in range(nx):
            f_vals[j, i] = f(x[i], y[j])
    
    factor = dx**2 * dy**2 / (2 * (dx**2 + dy**2))
    
    for iteration in range(max_iter):
        max_diff = 0.0
        
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                u_new = factor * (dy**2 * (u[j, i+1] + u[j, i-1]) + 
                                  dx**2 * (u[j+1, i] + u[j-1, i]) - 
                                  f_vals[j, i] * dx**2 * dy**2)
                
                # SOR update
                u_new = u[j, i] + omega * (u_new - u[j, i])
                
                diff = abs(u_new - u[j, i])
                if diff > max_diff:
                    max_diff = diff
                
                u[j, i] = u_new
        
        if max_diff < tol:
            print(f"Poisson converged after {iteration + 1} iterations, max_diff = {max_diff:.2e}")
            break
    else:
        print(f"Max iterations reached, max_diff = {max_diff:.2e}")
    
    return x, y, u


def optimal_omega(nx, ny):
    """
    Estimate optimal SOR relaxation parameter for Laplace on rectangle.
    """
    pi = np.pi
    return 2 / (1 + np.sin(pi / max(nx, ny)))


def poisson_2d_direct(nx, ny, Lx, Ly, f, bc_left, bc_right, bc_bottom, bc_top):
    """
    Solve 2D Poisson equation using direct method (sparse matrix).
    Only suitable for small grids due to memory.
    """
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    
    n_interior = (nx - 2) * (ny - 2)
    if n_interior > 10000:
        raise ValueError("Grid too large for direct method. Use iterative method.")
    
    A = np.zeros((n_interior, n_interior))
    b = np.zeros(n_interior)
    
    # Map (i,j) to index
    def idx(i, j):
        return (j - 1) * (nx - 2) + (i - 1)
    
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            k = idx(i, j)
            
            # Diagonal
            A[k, k] = -2/dx**2 - 2/dy**2
            
            # x neighbors
            if i > 1:
                A[k, idx(i-1, j)] = 1/dx**2
            else:
                b[k] -= bc_left(y[j]) / dx**2
            
            if i < nx - 2:
                A[k, idx(i+1, j)] = 1/dx**2
            else:
                b[k] -= bc_right(y[j]) / dx**2
            
            # y neighbors
            if j > 1:
                A[k, idx(i, j-1)] = 1/dy**2
            else:
                b[k] -= bc_bottom(x[i]) / dy**2
            
            if j < ny - 2:
                A[k, idx(i, j+1)] = 1/dy**2
            else:
                b[k] -= bc_top(x[i]) / dy**2
            
            # Source term
            b[k] += f(x[i], y[j])
    
    u_interior = np.linalg.solve(A, b)
    
    # Reconstruct full solution
    u = np.zeros((ny, nx))
    u[:, 0] = bc_left(y)
    u[:, -1] = bc_right(y)
    u[0, :] = bc_bottom(x)
    u[-1, :] = bc_top(x)
    
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            u[j, i] = u_interior[idx(i, j)]
    
    return x, y, u


def compute_error(u_num, u_exact):
    """Compute error norms."""
    error = u_num - u_exact
    l2 = np.sqrt(np.mean(error**2))
    linf = np.max(np.abs(error))
    return l2, linf


def main():
    print("=" * 60)
    print("2D Laplace & Poisson Equations")
    print("=" * 60)
    
    examples = {
        '1': {
            'name': 'Laplace: u=0 on 3 sides, u=sin(pi*x) on top',
            'type': 'laplace',
            'Lx': 1.0, 'Ly': 1.0,
            'bc_left': lambda y: 0,
            'bc_right': lambda y: 0,
            'bc_bottom': lambda x: 0,
            'bc_top': lambda x: np.sin(np.pi * x),
            'exact': lambda x, y: np.sin(np.pi * x) * np.sinh(np.pi * y) / np.sinh(np.pi)
        },
        '2': {
            'name': 'Poisson: ∇²u = -2π²sin(πx)sin(πy), zero BCs',
            'type': 'poisson',
            'Lx': 1.0, 'Ly': 1.0,
            'f': lambda x, y: -2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y),
            'bc_left': lambda y: 0,
            'bc_right': lambda y: 0,
            'bc_bottom': lambda x: 0,
            'bc_top': lambda x: 0,
            'exact': lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y)
        },
        '3': {
            'name': 'Poisson: ∇²u = -2, u=0 on boundaries (torsion problem)',
            'type': 'poisson',
            'Lx': 1.0, 'Ly': 1.0,
            'f': lambda x, y: -2,
            'bc_left': lambda y: 0,
            'bc_right': lambda y: 0,
            'bc_bottom': lambda x: 0,
            'bc_top': lambda x: 0,
            'exact': None  # No simple exact solution
        },
        '4': {
            'name': 'Custom',
            'type': None,
            'Lx': None, 'Ly': None,
            'f': None,
            'bc_left': None, 'bc_right': None,
            'bc_bottom': None, 'bc_top': None,
            'exact': None
        }
    }
    
    while True:
        print("\nExamples:")
        for k, v in examples.items():
            print(f"  {k}. {v['name']}")
        print("  5. Compare Jacobi vs Gauss-Seidel vs SOR")
        print("  6. Exit")
        
        choice = input("\nSelect (1-6): ")
        
        if choice == '6':
            break
        
        if choice == '5':
            # Comparison
            nx = ny = 31
            Lx = Ly = 1.0
            
            bc_left = lambda y: 0
            bc_right = lambda y: 0
            bc_bottom = lambda x: 0
            bc_top = lambda x: np.sin(np.pi * x)
            exact = lambda x, y: np.sin(np.pi * x) * np.sinh(np.pi * y) / np.sinh(np.pi)
            
            print("\nComparing methods...")
            
            # Jacobi
            x, y, u_j = laplace_2d_jacobi(nx, ny, Lx, Ly, bc_left, bc_right, bc_bottom, bc_top, 
                                          max_iter=5000, tol=1e-6)
            X, Y = np.meshgrid(x, y)
            l2_j, linf_j = compute_error(u_j, exact(X, Y))
            print(f"Jacobi:      L2={l2_j:.2e}, Linf={linf_j:.2e}")
            
            # Gauss-Seidel
            x, y, u_gs = laplace_2d_gs(nx, ny, Lx, Ly, bc_left, bc_right, bc_bottom, bc_top, 
                                        max_iter=5000, tol=1e-6, omega=1.0)
            l2_gs, linf_gs = compute_error(u_gs, exact(X, Y))
            print(f"Gauss-Seidel: L2={l2_gs:.2e}, Linf={linf_gs:.2e}")
            
            # SOR with optimal omega
            omega_opt = optimal_omega(nx, ny)
            x, y, u_sor = laplace_2d_gs(nx, ny, Lx, Ly, bc_left, bc_right, bc_bottom, bc_top, 
                                         max_iter=5000, tol=1e-6, omega=omega_opt)
            l2_sor, linf_sor = compute_error(u_sor, exact(X, Y))
            print(f"SOR (ω={omega_opt:.4f}): L2={l2_sor:.2e}, Linf={linf_sor:.2e}")
            
            continue
        
        if choice not in examples:
            print("Invalid!")
            continue
        
        ex = examples[choice]
        
        if choice == '4':
            ex['type'] = input("Type (laplace/poisson): ")
            ex['Lx'] = float(input("Lx: "))
            ex['Ly'] = float(input("Ly: "))
            if ex['type'] == 'poisson':
                f_str = input("f(x,y) = ")
                ex['f'] = lambda x, y: eval(f_str, {"np": np, "x": x, "y": y})
            bc_str = input("bc_left(y) = ")
            ex['bc_left'] = lambda y: eval(bc_str, {"np": np, "y": y})
            bc_str = input("bc_right(y) = ")
            ex['bc_right'] = lambda y: eval(bc_str, {"np": np, "y": y})
            bc_str = input("bc_bottom(x) = ")
            ex['bc_bottom'] = lambda x: eval(bc_str, {"np": np, "x": x})
            bc_str = input("bc_top(x) = ")
            ex['bc_top'] = lambda x: eval(bc_str, {"np": np, "x": x})
        
        nx = int(input("\nnx: "))
        ny = int(input("ny: "))
        
        if ex['type'] == 'laplace':
            print(f"\nOptimal SOR omega: {optimal_omega(nx, ny):.4f}")
            omega = float(input(f"Relaxation omega (1.0=GS, optimal≈{optimal_omega(nx, ny):.4f}): ") or optimal_omega(nx, ny))
            
            x, y, u = laplace_2d_gs(nx, ny, ex['Lx'], ex['Ly'], 
                                     ex['bc_left'], ex['bc_right'], 
                                     ex['bc_bottom'], ex['bc_top'],
                                     omega=omega)
            
        else:  # poisson
            print(f"\nOptimal SOR omega: {optimal_omega(nx, ny):.4f}")
            omega = float(input(f"Relaxation omega (1.0=GS, optimal≈{optimal_omega(nx, ny):.4f}): ") or optimal_omega(nx, ny))
            
            x, y, u = poisson_2d_gs(nx, ny, ex['Lx'], ex['Ly'], ex['f'],
                                     ex['bc_left'], ex['bc_right'], 
                                     ex['bc_bottom'], ex['bc_top'],
                                     omega=omega)
            
            # Also try direct method for small grids
            if nx <= 21 and ny <= 21:
                x_d, y_d, u_d = poisson_2d_direct(nx, ny, ex['Lx'], ex['Ly'], ex['f'],
                                                   ex['bc_left'], ex['bc_right'], 
                                                   ex['bc_bottom'], ex['bc_top'])
                print("Direct method: Done")
                X, Y = np.meshgrid(x, y)
                l2, linf = compute_error(u, u_d)
                print(f"Diff from direct: L2={l2:.2e}, Linf={linf:.2e}")
        
        # Error analysis if exact solution available
        if ex['exact']:
            X, Y = np.meshgrid(x, y)
            u_exact = ex['exact'](X, Y)
            l2, linf = compute_error(u, u_exact)
            print(f"\nError norms: L2 = {l2:.2e}, L∞ = {linf:.2e}")
            
            # Print cross-section
            mid_y = ny // 2
            print(f"\nCross-section at y = {y[mid_y]:.3f}:")
            print(f"{'x':<8} {'Numerical':<12} {'Exact':<12} {'Error':<10}")
            print("-" * 42)
            for i in range(0, nx, max(1, nx//10)):
                print(f"{x[i]:<8.4f} {u[mid_y, i]:<12.6f} {u_exact[mid_y, i]:<12.6f} {abs(u[mid_y, i] - u_exact[mid_y, i]):<10.2e}")


if __name__ == "__main__":
    main()