import numpy as np
import matplotlib.pyplot as plt


def heat_explicit(alpha, L, T, nx, nt, u0, u_left, u_right):
    """
    Solve 1D heat equation using explicit FTCS method:
    ∂u/∂t = alpha * ∂²u/∂x²
    
    Parameters:
    alpha : thermal diffusivity
    L : length of rod
    T : total time
    nx : number of spatial grid points
    nt : number of time steps
    u0 : initial condition function u0(x)
    u_left, u_right : boundary condition functions of time
    
    Returns:
    x, t, u : grid and solution array (nt+1 x nx)
    """
    dx = L / (nx - 1)
    dt = T / nt
    r = alpha * dt / dx**2
    
    if r > 0.5:
        print(f"Warning: r = {r:.4f} > 0.5, explicit method may be unstable!")
    
    x = np.linspace(0, L, nx)
    t = np.linspace(0, T, nt + 1)
    u = np.zeros((nt + 1, nx))
    
    # Initial condition
    u[0, :] = u0(x)
    
    # Boundary conditions
    for n in range(nt + 1):
        u[n, 0] = u_left(t[n])
        u[n, -1] = u_right(t[n])
    
    # Time stepping
    for n in range(nt):
        for i in range(1, nx - 1):
            u[n+1, i] = u[n, i] + r * (u[n, i+1] - 2*u[n, i] + u[n, i-1])
    
    return x, t, u


def heat_implicit(alpha, L, T, nx, nt, u0, u_left, u_right):
    """
    Solve 1D heat equation using fully implicit (backward Euler) method.
    Unconditionally stable.
    """
    dx = L / (nx - 1)
    dt = T / nt
    r = alpha * dt / dx**2
    
    x = np.linspace(0, L, nx)
    t = np.linspace(0, T, nt + 1)
    u = np.zeros((nt + 1, nx))
    
    u[0, :] = u0(x)
    for n in range(nt + 1):
        u[n, 0] = u_left(t[n])
        u[n, -1] = u_right(t[n])
    
    # Tridiagonal matrix for implicit scheme
    # (1+2r)u_i^{n+1} - r*u_{i-1}^{n+1} - r*u_{i+1}^{n+1} = u_i^n
    n_interior = nx - 2
    A = np.zeros((n_interior, n_interior))
    for i in range(n_interior):
        A[i, i] = 1 + 2*r
        if i > 0:
            A[i, i-1] = -r
        if i < n_interior - 1:
            A[i, i+1] = -r
    
    # Time stepping
    for n in range(nt):
        # Right-hand side
        b = u[n, 1:-1].copy()
        # Boundary contributions
        b[0] += r * u_left(t[n+1])
        b[-1] += r * u_right(t[n+1])
        
        # Solve
        u[n+1, 1:-1] = np.linalg.solve(A, b)
    
    return x, t, u


def heat_crank_nicolson(alpha, L, T, nx, nt, u0, u_left, u_right):
    """
    Solve 1D heat equation using Crank-Nicolson method.
    Second-order accurate in both space and time, unconditionally stable.
    """
    dx = L / (nx - 1)
    dt = T / nt
    r = alpha * dt / dx**2
    
    x = np.linspace(0, L, nx)
    t = np.linspace(0, T, nt + 1)
    u = np.zeros((nt + 1, nx))
    
    u[0, :] = u0(x)
    for n in range(nt + 1):
        u[n, 0] = u_left(t[n])
        u[n, -1] = u_right(t[n])
    
    # Matrices for Crank-Nicolson:
    # (I + r/2 * A) u^{n+1} = (I - r/2 * A) u^n + boundary terms
    n_interior = nx - 2
    
    A = np.zeros((n_interior, n_interior))
    B = np.zeros((n_interior, n_interior))
    
    for i in range(n_interior):
        A[i, i] = 1 + r
        B[i, i] = 1 - r
        if i > 0:
            A[i, i-1] = -r/2
            B[i, i-1] = r/2
        if i < n_interior - 1:
            A[i, i+1] = -r/2
            B[i, i+1] = r/2
    
    # Pre-factorize A for efficiency
    # Using numpy's solve (LU decomposition internally)
    
    for n in range(nt):
        # Right-hand side
        b = B @ u[n, 1:-1]
        # Boundary contributions
        b[0] += (r/2) * (u_left(t[n]) + u_left(t[n+1]))
        b[-1] += (r/2) * (u_right(t[n]) + u_right(t[n+1]))
        
        u[n+1, 1:-1] = np.linalg.solve(A, b)
    
    return x, t, u


def heat_crank_nicolson_thomas(alpha, L, T, nx, nt, u0, u_left, u_right):
    """
    Crank-Nicolson using Thomas algorithm (more efficient for tridiagonal).
    """
    dx = L / (nx - 1)
    dt = T / nt
    r = alpha * dt / dx**2
    
    x = np.linspace(0, L, nx)
    t = np.linspace(0, T, nt + 1)
    u = np.zeros((nt + 1, nx))
    
    u[0, :] = u0(x)
    for n in range(nt + 1):
        u[n, 0] = u_left(t[n])
        u[n, -1] = u_right(t[n])
    
    n_interior = nx - 2
    
    # Thomas algorithm coefficients (constant for Crank-Nicolson)
    a = -r/2 * np.ones(n_interior - 1)  # sub-diagonal
    b = (1 + r) * np.ones(n_interior)    # diagonal
    c = -r/2 * np.ones(n_interior - 1)   # super-diagonal
    
    def thomas_solve(a, b, c, d):
        n = len(b)
        cp = np.zeros(n-1)
        dp = np.zeros(n)
        
        cp[0] = c[0] / b[0]
        dp[0] = d[0] / b[0]
        
        for i in range(1, n-1):
            denom = b[i] - a[i-1] * cp[i-1]
            cp[i] = c[i] / denom
            dp[i] = (d[i] - a[i-1] * dp[i-1]) / denom
        
        dp[n-1] = (d[n-1] - a[n-2] * dp[n-2]) / (b[n-1] - a[n-2] * cp[n-2])
        
        x = np.zeros(n)
        x[n-1] = dp[n-1]
        for i in range(n-2, -1, -1):
            x[i] = dp[i] - cp[i] * x[i+1]
        return x
    
    for n in range(nt):
        # RHS: (I - r/2 A) u^n + boundary terms
        d = np.zeros(n_interior)
        for i in range(n_interior):
            d[i] = (1 - r) * u[n, i+1]
            if i > 0:
                d[i] += (r/2) * u[n, i]
            if i < n_interior - 1:
                d[i] += (r/2) * u[n, i+2]
        
        # Boundary contributions
        d[0] += (r/2) * (u_left(t[n]) + u_left(t[n+1]))
        d[-1] += (r/2) * (u_right(t[n]) + u_right(t[n+1]))
        
        u[n+1, 1:-1] = thomas_solve(a, b, c, d)
    
    return x, t, u


def heat_with_source_explicit(alpha, L, T, nx, nt, u0, u_left, u_right, f):
    """
    Heat equation with source term: ∂u/∂t = alpha ∂²u/∂x² + f(x,t)
    """
    dx = L / (nx - 1)
    dt = T / nt
    r = alpha * dt / dx**2
    
    x = np.linspace(0, L, nx)
    t = np.linspace(0, T, nt + 1)
    u = np.zeros((nt + 1, nx))
    
    u[0, :] = u0(x)
    for n in range(nt + 1):
        u[n, 0] = u_left(t[n])
        u[n, -1] = u_right(t[n])
    
    for n in range(nt):
        for i in range(1, nx - 1):
            u[n+1, i] = (u[n, i] + r * (u[n, i+1] - 2*u[n, i] + u[n, i-1]) 
                         + dt * f(x[i], t[n]))
    
    return x, t, u


def heat_crank_nicolson_source(alpha, L, T, nx, nt, u0, u_left, u_right, f):
    """
    Crank-Nicolson with source term.
    """
    dx = L / (nx - 1)
    dt = T / nt
    r = alpha * dt / dx**2
    
    x = np.linspace(0, L, nx)
    t = np.linspace(0, T, nt + 1)
    u = np.zeros((nt + 1, nx))
    
    u[0, :] = u0(x)
    for n in range(nt + 1):
        u[n, 0] = u_left(t[n])
        u[n, -1] = u_right(t[n])
    
    n_interior = nx - 2
    a = -r/2 * np.ones(n_interior - 1)
    b = (1 + r) * np.ones(n_interior)
    c = -r/2 * np.ones(n_interior - 1)
    
    def thomas_solve(a, b, c, d):
        n = len(b)
        cp = np.zeros(n-1)
        dp = np.zeros(n)
        cp[0] = c[0] / b[0]
        dp[0] = d[0] / b[0]
        for i in range(1, n-1):
            denom = b[i] - a[i-1] * cp[i-1]
            cp[i] = c[i] / denom
            dp[i] = (d[i] - a[i-1] * dp[i-1]) / denom
        dp[n-1] = (d[n-1] - a[n-2] * dp[n-2]) / (b[n-1] - a[n-2] * cp[n-2])
        x = np.zeros(n)
        x[n-1] = dp[n-1]
        for i in range(n-2, -1, -1):
            x[i] = dp[i] - cp[i] * x[i+1]
        return x
    
    for n in range(nt):
        d = np.zeros(n_interior)
        for i in range(n_interior):
            xi = x[i+1]
            d[i] = (1 - r) * u[n, i+1] + dt/2 * (f(xi, t[n]) + f(xi, t[n+1]))
            if i > 0:
                d[i] += (r/2) * u[n, i]
            if i < n_interior - 1:
                d[i] += (r/2) * u[n, i+2]
        
        d[0] += (r/2) * (u_left(t[n]) + u_left(t[n+1]))
        d[-1] += (r/2) * (u_right(t[n]) + u_right(t[n+1]))
        
        u[n+1, 1:-1] = thomas_solve(a, b, c, d)
    
    return x, t, u


def plot_solution(x, t, u, title="Heat Equation Solution", save_path=None):
    """Plot heat equation solution."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 3D surface
    X, T = np.meshgrid(x, t)
    ax = axes[0]
    im = ax.contourf(X, T, u, levels=20, cmap='hot')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    
    # Profile at different times
    ax = axes[1]
    n_plots = min(5, len(t))
    indices = np.linspace(0, len(t)-1, n_plots, dtype=int)
    for idx in indices:
        ax.plot(x, u[idx], label=f't={t[idx]:.2f}')
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.set_title('Temperature Profiles')
    ax.legend()
    ax.grid(True)
    
    # Temperature at center vs time
    ax = axes[2]
    center_idx = len(x) // 2
    ax.plot(t, u[:, center_idx], 'b-', linewidth=2)
    ax.set_xlabel('t')
    ax.set_ylabel('u(center)')
    ax.set_title('Center Temperature vs Time')
    ax.grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def compare_methods():
    """Compare explicit, implicit, and Crank-Nicolson."""
    alpha = 0.01
    L = 1.0
    T = 0.5
    nx = 21
    
    # Test with different time steps
    for nt in [50, 250, 500]:
        dt = T / nt
        r = alpha * dt / (L/(nx-1))**2
        
        u0 = lambda x: np.sin(np.pi * x)
        u_left = lambda t: 0
        u_right = lambda t: 0
        exact = lambda x, t: np.exp(-alpha * np.pi**2 * t) * np.sin(np.pi * x)
        
        print(f"\n{'='*60}")
        print(f"nt={nt}, dt={dt:.6f}, r={r:.4f}")
        print(f"{'='*60}")
        
        # Explicit
        if r <= 0.5:
            x, t, u_exp = heat_explicit(alpha, L, T, nx, nt, u0, u_left, u_right)
            err_exp = np.max(np.abs(u_exp - exact(x, t[:, None])))
            print(f"Explicit FTCS max error: {err_exp:.6f}")
        else:
            print("Explicit FTCS: UNSTABLE (r > 0.5)")
        
        # Implicit
        x, t, u_imp = heat_implicit(alpha, L, T, nx, nt, u0, u_left, u_right)
        err_imp = np.max(np.abs(u_imp - exact(x, t[:, None])))
        print(f"Implicit (Backward Euler) max error: {err_imp:.6f}")
        
        # Crank-Nicolson
        x, t, u_cn = heat_crank_nicolson(alpha, L, T, nx, nt, u0, u_left, u_right)
        err_cn = np.max(np.abs(u_cn - exact(x, t[:, None])))
        print(f"Crank-Nicolson max error: {err_cn:.6f}")


def main():
    print("=" * 60)
    print("1D Heat Equation: Explicit, Implicit, Crank-Nicolson")
    print("=" * 60)
    
    examples = {
        '1': {
            'name': 'Standard: u_t = alpha*u_xx, u(x,0)=sin(pi*x), u(0,t)=u(1,t)=0',
            'alpha': 0.01,
            'L': 1.0,
            'u0': lambda x: np.sin(np.pi * x),
            'u_left': lambda t: 0,
            'u_right': lambda t: 0,
            'exact': lambda x, t: np.exp(-0.01 * np.pi**2 * t) * np.sin(np.pi * x)
        },
        '2': {
            'name': 'With source: u_t = alpha*u_xx + sin(pi*x)*exp(-t), zero BCs',
            'alpha': 0.01,
            'L': 1.0,
            'u0': lambda x: 0,
            'u_left': lambda t: 0,
            'u_right': lambda t: 0,
            'f': lambda x, t: np.sin(np.pi * x) * np.exp(-t),
            'exact': None
        },
        '3': {
            'name': 'Non-zero BCs: u_t = alpha*u_xx, u(x,0)=0, u(0,t)=1, u(1,t)=0',
            'alpha': 0.01,
            'L': 1.0,
            'u0': lambda x: 0,
            'u_left': lambda t: 1,
            'u_right': lambda t: 0,
            'exact': None
        },
        '4': {
            'name': 'Custom problem',
            'alpha': None, 'L': None, 'T': None,
            'u0': None, 'u_left': None, 'u_right': None,
            'f': None, 'exact': None
        }
    }
    
    while True:
        print("\nExamples:")
        for k, v in examples.items():
            print(f"  {k}. {v['name']}")
        print("  5. Compare methods (stability demo)")
        print("  6. Exit")
        
        choice = input("\nSelect (1-6): ")
        
        if choice == '6':
            break
        
        if choice == '5':
            compare_methods()
            continue
        
        if choice not in examples:
            print("Invalid!")
            continue
        
        ex = examples[choice]
        
        if choice == '4':
            ex['alpha'] = float(input("alpha: "))
            ex['L'] = float(input("L: "))
            ex['T'] = float(input("T: "))
            u0_str = input("u0(x) = ")
            ex['u0'] = lambda x: eval(u0_str, {"np": np, "x": x})
            ex['u_left'] = lambda t: eval(input("u(0,t) = "), {"np": np, "t": t})
            ex['u_right'] = lambda t: eval(input("u(L,t) = "), {"np": np, "t": t})
            f_str = input("Source f(x,t) (optional): ")
            if f_str:
                ex['f'] = lambda x, t: eval(f_str, {"np": np, "x": x, "t": t})
        
        nx = int(input("\nnx (spatial points): "))
        nt = int(input("nt (time steps): "))
        T = ex.get('T', float(input("T (final time): ")))
        
        print(f"\n{'='*60}")
        print(f"Solving: {ex['name']}")
        print(f"alpha={ex['alpha']}, L={ex['L']}, T={T}")
        print(f"nx={nx}, nt={nt}")
        dx = ex['L'] / (nx - 1)
        dt = T / nt
        r = ex['alpha'] * dt / dx**2
        print(f"dx={dx:.4f}, dt={dt:.6f}, r={r:.4f}")
        print(f"{'='*60}")
        
        # Explicit
        if r <= 0.5:
            x, t, u_exp = heat_explicit(ex['alpha'], ex['L'], T, nx, nt, 
                                         ex['u0'], ex['u_left'], ex['u_right'])
            print("Explicit FTCS: Done")
        else:
            print(f"Explicit FTCS: SKIPPED (r={r:.4f} > 0.5, unstable)")
            u_exp = None
        
        # Implicit
        x, t, u_imp = heat_implicit(ex['alpha'], ex['L'], T, nx, nt, 
                                     ex['u0'], ex['u_left'], ex['u_right'])
        print("Implicit (Backward Euler): Done")
        
        # Crank-Nicolson
        x, t, u_cn = heat_crank_nicolson(ex['alpha'], ex['L'], T, nx, nt, 
                                          ex['u0'], ex['u_left'], ex['u_right'])
        print("Crank-Nicolson: Done")
        
        # With source if provided
        if ex.get('f'):
            x, t, u_exp_src = heat_with_source_explicit(ex['alpha'], ex['L'], T, nx, nt,
                                                         ex['u0'], ex['u_left'], ex['u_right'], ex['f'])
            x, t, u_cn_src = heat_crank_nicolson_source(ex['alpha'], ex['L'], T, nx, nt,
                                                         ex['u0'], ex['u_left'], ex['u_right'], ex['f'])
            print("With source term: Done")
        
        # Compare at center
        center = nx // 2
        print(f"\nCenter point (x={x[center]:.3f}) at t={T}:")
        if u_exp is not None:
            print(f"  Explicit:     {u_exp[-1, center]:.6f}")
        print(f"  Implicit:     {u_imp[-1, center]:.6f}")
        print(f"  Crank-Nicols: {u_cn[-1, center]:.6f}")
        
        if ex['exact']:
            exact_center = ex['exact'](x[center], T)
            print(f"  Exact:        {exact_center:.6f}")
            if u_exp is not None:
                print(f"  Explicit err: {abs(u_exp[-1, center] - exact_center):.2e}")
            print(f"  Implicit err: {abs(u_imp[-1, center] - exact_center):.2e}")
            print(f"  CN error:     {abs(u_cn[-1, center] - exact_center):.2e}")
        
        # Plot option
        if input("\nPlot results? (y/n): ").lower() == 'y':
            if u_exp is not None:
                plot_solution(x, t, u_exp, "Explicit FTCS")
            plot_solution(x, t, u_imp, "Implicit (Backward Euler)")
            plot_solution(x, t, u_cn, "Crank-Nicolson")


if __name__ == "__main__":
    main()