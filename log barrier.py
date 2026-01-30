import numpy as np

def f(x,y):
    return 2*x**2 + 3*y**2 - x*y

def grad_f(x,y):
    return np.array([4*x - y, 6*y - x], dtype=float)

def g(x,y):
    return np.log(x-1) + np.log(2-x) + np.log(y-1) + np.log(2-y)

def grad_g(x,y):
    return np.array([1/(x-1) - 1/(2-x), 1/(y-1) - 1/(2-y)], dtype=float)

def phi(x,y, mu):
    return f(x,y) - mu * g(x,y)

def grad_phi(x, y, mu):
    return grad_f(x,y) - mu * grad_g(x,y)

def in_interior(x,y, eps=1e-12):
    return (1+eps < x < 2-eps) and (1+eps < y < 2-eps)

def unconstrained_grad_descent_interior(x0, mu, tol=1e-6, max_iter=1000, armijo_c=1e-4):
    if not in_interior(x0[0], x0[1]):
        raise ValueError("Initial point is not in the interior of the feasible region.")
    x_old, y_old = x0[0], x0[1]
    alpha0 = 1.0
    beta = 0.5   # backtracking shrink

    for k in range(max_iter):

        # gradient
        gx, gy = grad_phi(x_old, y_old, mu)
        grad = np.array([gx, gy])
        gnorm = np.linalg.norm(grad)

        if gnorm < tol:
            return np.array([x_old, y_old], dtype=float)

        # descent direction
        d = -grad
        
        alpha = alpha0
        phi_old = phi(x_old, y_old, mu)

        # Armijo + interior backtracking
        while True:
            x_new = x_old + alpha * d[0]
            y_new = y_old + alpha * d[1]

            if not in_interior(x_new, y_new):
                alpha *= beta
                continue

            phi_new = phi(x_new, y_new, mu)

            if phi_new <= phi_old + armijo_c * alpha * (grad @ d): # Armijo condition
                break

            alpha *= beta

            if alpha < 1e-14:
                raise RuntimeError("Step size collapsed")

        # update
        if np.linalg.norm([x_new-x_old, y_new-y_old]) < tol:
            return np.array([x_new, y_new], dtype=float)

        x_old, y_old = x_new, y_new

    return np.array([x_old, y_old], dtype=float)

def log_barrier_method(x0, mu0=0.5, tau=0.1,
                       outer_tol=1e-10, inner_tol=1e-8,
                       max_outer=50):
    x = np.array(x0, dtype=float)
    if not in_interior(x[0], x[1]):
        raise ValueError("x0 must be strictly inside the feasible set.")

    mu = mu0
    m = 4  # number of inequality constraints in (1,2)^2

    for k in range(max_outer):
        # Solve min phi_mu(x) approximately
        x = unconstrained_grad_descent_interior(x, mu, tol=inner_tol)

        # duality-gap style stop
        if m * mu <= outer_tol:
            break

        mu *= tau

    return x, f(x[0], x[1])

if __name__ == "__main__":
    x0 = np.array([1.5, 1.5], dtype=float)
    x_opt, f_opt = log_barrier_method(x0)
    print(f"Optimal point: {x_opt}, Optimal value: {f_opt}")