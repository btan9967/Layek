import numpy as np
import scipy.integrate as sp
import matplotlib.pyplot as plt

# Parameters
c = 1.0  # positive constant
N = 1000  # number of points for plotting

# Time array
t = np.linspace(0, 10, N)

# Function to compute the derivatives
def derivatives(state, t, c):
    r, theta = state
    drdt = c * r * (1 - r)
    dthetadt = 1
    return [drdt, dthetadt]

# Initial conditions and solving the ODEs for different starting radii
radii = np.linspace(0.1, 1.5, 5)
for r0 in radii:
    sol = sp.odeint(derivatives, [r0, 0], t, args=(c,))

    # Convert to Cartesian coordinates for plotting
    x = sol[:, 0] * np.cos(sol[:, 1])
    y = sol[:, 0] * np.sin(sol[:, 1])

    plt.plot(x, y, label=f'r0 = {r0:.1f}')

# Adding the limit cycle (r = 1 circle)
theta = np.linspace(0, 2*np.pi, 100)
x_limit_cycle = np.cos(theta)
y_limit_cycle = np.sin(theta)
plt.plot(x_limit_cycle, y_limit_cycle, 'k--', label='Limit Cycle (r = 1)')

# Plot settings
plt.xlabel('x')
plt.ylabel('y')
plt.title('Trajectories in the Dynamical System')
plt.legend(loc = 'lower right')
plt.axis('equal')
plt.grid(True)
plt.show()
