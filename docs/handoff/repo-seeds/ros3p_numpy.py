"""Independent ROS3P (Lang & Verwer 2001, transformed form) one-step check."""
import numpy as np
from scipy.integrate import solve_ivp
G, W, D = 1.0, 1.0, 0.0
def f(y):
    u, v, w = y
    return np.array([-0.5*G*u + D*v, -D*u - 0.5*G*v - W*w, W*v - G*(w + 1.0)])
J = np.array([[-0.5*G, D, 0.0], [-D, -0.5*G, -W], [0.0, W, -G]])
gamma = 0.5 + np.sqrt(3)/6; ig = 1/gamma
a = np.array([[0,0,0],[ig,0,0],[ig,0,0]])
c = np.array([[0,0,0],[-ig**2,0,0],[-ig*(1+ig*(2-0.5*ig)), -ig*(2-0.5*ig), 0]])
m = np.array([2.0, ig*(2/3 - ig/6), ig/3])
mhat = np.array([2.113248654051871, 1.0, 0.4226497308103742])
def ros3p_step(y, h):
    A = np.eye(3)/(h*gamma) - J
    u = []
    for i in range(3):
        ys = y + sum(a[i,j]*u[j] for j in range(i))
        rhs = f(ys) + sum((c[i,j]/h)*u[j] for j in range(i))
        u.append(np.linalg.solve(A, rhs))
    u = np.array(u)
    y_new = y + m @ u
    y_hat = y + mhat @ u
    return y_new, y_new - y_hat
y0 = np.array([0.0, 0.0, -1.0])
for h in [0.5, 0.25, 0.125, 0.0625]:
    y_new, est = ros3p_step(y0, h)
    ref = solve_ivp(lambda t, y: f(y), (0, h), y0, method="DOP853", rtol=1e-13, atol=1e-15).y[:, -1]
    print(f"h={h:7.4f}  true local err={np.max(np.abs(y_new-ref)):.3e}  (m-mhat).u estimate={np.max(np.abs(est)):.3e}  ratio={np.max(np.abs(est))/np.max(np.abs(y_new-ref)):.2f}")
