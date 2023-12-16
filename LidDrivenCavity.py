import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 1.0         # Length of the cavity (1x1 square)
N = 50          # Number of grid points in each direction
U = 1.0         # Lid velocity
nu = 0.1        # Kinematic viscosity
dx = L / (N-1)  # Grid spacing
dy = dx
dt = 0.001      # Time step

# Initialize the grid
x = np.linspace(0, L, N)
y = np.linspace(0, L, N)
X, Y = np.meshgrid(x, y)

# Initialize vorticity and stream function
omega = np.zeros((N, N))
psi = np.zeros((N, N))

def laplacian(Z):
    """Calculate the Laplacian of a 2D array."""
    Ztop = Z[:-2, 1:-1]
    Zleft = Z[1:-1, :-2]
    Zbottom = Z[2:, 1:-1]
    Zright = Z[1:-1, 2:]
    Zcenter = Z[1:-1, 1:-1]
    return (Ztop + Zleft + Zbottom + Zright - 4 * Zcenter) / dx**2

def update_stream_function(psi, omega):
    """Solve the Poisson equation for the stream function."""
    for _ in range(1000):  # Iterations for Poisson solver
        psi[1:-1, 1:-1] = 0.25 * (psi[:-2, 1:-1] + psi[1:-1, :-2] + psi[1:-1, 2:] + psi[2:, 1:-1] + omega[1:-1, 1:-1] * dx**2)
    return psi

def update_vorticity(omega, psi):
    """Update the vorticity field."""
    omega[1:-1, 1:-1] = omega[1:-1, 1:-1] - dt * (
        (psi[1:-1, 2:] - psi[1:-1, :-2]) * (omega[2:, 1:-1] - omega[:-2, 1:-1]) / (4 * dy**2) -
        (psi[2:, 1:-1] - psi[:-2, 1:-1]) * (omega[1:-1, 2:] - omega[1:-1, :-2]) / (4 * dx**2)
    ) + nu * dt * laplacian(omega)
    return omega

# Main loop
for t in range(1000):
    # Boundary conditions for vorticity
    omega[-1, :] = -2 * (psi[-2, :] - U * dy) / dy**2  # Top boundary (lid)
    omega[0, :] = -2 * psi[1, :] / dy**2               # Bottom boundary
    omega[:, 0] = -2 * psi[:, 1] / dx**2               # Left boundary
    omega[:, -1] = -2 * psi[:, -2] / dx**2             # Right boundary

    # Update vorticity and stream function
    omega = update_vorticity(omega, psi)
    psi = update_stream_function(psi, omega)

# Plotting
plt.contourf(X, Y, psi, cmap='viridis', levels=50)
plt.colorbar(label='Stream Function')
plt.streamplot(X, Y, np.gradient(psi, axis=1), -np.gradient(psi, axis=0), color='black')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Lid-Driven Cavity Flow')
plt.show()
