import numpy as np

def f(x, y):
    return 2*x**2 + 3*y**2 - x*y

def grad_f(x, y):
    return np.array([4*x - y, 6*y - x], dtype=float)

def projection_onto_feasible_set(z, lower_bound = 1, upper_bound = 2):
    return np.clip(z, lower_bound, upper_bound)

def projected_gd(x0, tol=1e-10, max_iter=1000):
    x = np.array(x0, dtype=float)
    alpha = 0.1  # step size
    for k in range(max_iter):
        g = grad_f(x[0], x[1])
        y_next = x - alpha * g #gradient descent            
        x_next = projection_onto_feasible_set(y_next)    # projection

        if np.linalg.norm(x_next - x) < tol:
            return x_next, f(x_next[0], x_next[1]), k+1

        x = x_next

    return x, f(x[0], x[1]), max_iter

if __name__ == "__main__":
    X,Y = np.meshgrid(np.arange(1,2,0.01), np.arange(1,2,0.01))
    points = np.column_stack([X.ravel(), Y.ravel()])
    #Checking through a meshgrid of points in the feasible set. Hopefully this
    #satisfies the definition of "experimental verification".

    for i in points:
        sol, val, iters = projected_gd(i)
        print(f"Starting point: {i}, Solution: {sol}, Function value: {val}, Iterations: {iters}")

