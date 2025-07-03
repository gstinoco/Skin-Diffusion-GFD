# =============================================================================
# Skin Diffusion Simulator with Generalized Finite Differences (GFD)
# =============================================================================

'''
This module implements a skin diffusion simulator using the Generalized Finite 
Differences (GFD) method. It allows solving the diffusion equation on unstructured 
meshes with high precision and computational efficiency.

Main features:
- Optimized implementation with Numba JIT for intensive calculations
- Support for unstructured meshes
- Automatic calculation of GFD coefficients
- Numerical stability verification (CFL condition)
- Results visualization

All the codes presented below were developed by:
    Dr. Gerardo Tinoco Guerrero
    Universidad Michoacana de San Nicolás de Hidalgo
    gerardo.tinoco@umich.mx

With the funding of:
    Secretary of Science, Humanities, Technology and Innovation, SECIHTI (Secretaria de Ciencia, Humanidades, Tecnología e Innovación). México.
    Coordination of Scientific Research, CIC-UMSNH (Coordinación de la Investigación Científica de la Universidad Michoacana de San Nicolás de Hidalgo, CIC-UMSNH). México
    Aula CIMNE-Morelia. México
    SIIIA-MATH: Soluciones de Ingeniería. México

Date:
    June, 2025.

Last Modification:
    July, 2025.
'''

# Required packages for the solution
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import numba

@numba.jit(nopython = True)
def difusion_skin_jit(m, n, t, dt, nu, u_init, Gamma):
    '''
    Optimized version with Numba JIT of the diffusion function.
    Solves the diffusion equation on a two-dimensional mesh with Dirichlet conditions at the inlet
    and Neumann conditions at the lateral boundaries.
    
    Parameters:
        m (int):        Number of nodes in x direction.
        n (int):        Number of nodes in y direction.
        t (int):        Number of time steps.
        dt (float):     Time step size.
        nu (float):     Diffusion coefficient.
        u_init (float): Initial concentration value at the inlet.
        Gamma (array):  Pre-calculated GFD coefficients.
    Returns:
        u_final (array): Final solution of the diffusion equation on the mesh.
    '''
    # Initialize variables with float64 precision
    u_prev = np.zeros((m, n), dtype=np.float64)                                     # Previous state (float64)
    u_curr = np.zeros((m, n), dtype=np.float64)                                     # Current state (float64)
    Gamma_scaled = Gamma * nu * dt                                                  # Scaled Gammas with scientific precision (float64)

    # Initial condition
    u_prev[:, :] = 0.0                                                              # Initial Condition
    
    # Solve the differential equation (JIT optimized loop)
    for k in range(1, t):                                                           # Iterate over time steps
        # Update internal nodes using optimized einsum
        for i in range(1, m-1):                                                     # Iterate over internal nodes in x
            for j in range(1, n-1):                                                 # Iterate over internal nodes in y
                # Calculate diffusion contribution using Gamma coefficients
                diffusion = (
                    Gamma_scaled[i, j, 0] * u_prev[i, j] +                          # Center
                    Gamma_scaled[i, j, 1] * u_prev[i+1, j] +                        # East
                    Gamma_scaled[i, j, 2] * u_prev[i+1, j+1] +                      # Northeast
                    Gamma_scaled[i, j, 3] * u_prev[i, j+1] +                        # North
                    Gamma_scaled[i, j, 4] * u_prev[i-1, j+1] +                      # Northwest
                    Gamma_scaled[i, j, 5] * u_prev[i-1, j] +                        # West
                    Gamma_scaled[i, j, 6] * u_prev[i-1, j-1] +                      # Southwest
                    Gamma_scaled[i, j, 7] * u_prev[i, j-1] +                        # South
                    Gamma_scaled[i, j, 8] * u_prev[i+1, j-1]                        # Southeast
                )
                u_curr[i, j] = u_prev[i, j] + diffusion
        
        # Boundary conditions
        for i in range(m):
            u_curr[i, 0]   = u_init                                                 # Dirichlet at inlet (j = 0)
        
        for i in range(m):
            u_curr[i, n-1] = u_curr[i, n-2]                                         # Neumann at outlet (j = n-1)
        
        for j in range(n):
            u_curr[m-1, j] = u_curr[m-2, j]                                         # Neumann at left boundary (i = m-1)
        
        for j in range(n):
            u_curr[0, j]   = u_curr[1, j]                                           # Neumann at right boundary (i = 0)
        
        # Swap matrices for next step
        u_prev, u_curr     = u_curr, u_prev                                         # Optimized swap
    
    return u_prev                                                                   # u_prev contains the final solution

def difusion_skin(m, n, T, nu, u_init, Gamma):
    '''
    Solves the diffusion equation on a two-dimensional mesh with Dirichlet conditions at the inlet
    and Neumann conditions at the lateral boundaries.
    
    Parameters:
        m (int):        Number of nodes in x direction.
        n (int):        Number of nodes in y direction.
        T (array):      Time vector.
        nu (float):     Diffusion coefficient.
        u_init (float): Initial concentration value at the inlet.
        Gamma (array):  Pre-calculated GFD coefficients.
    Returns:
        u_final (array): Final solution of the diffusion equation on the mesh.
    '''
    # Initialize variables with float64 precision
    t      = len(T)                                                                 # Number of time steps
    dt     = T[1] - T[0]                                                            # Time step size (float64)
    u_prev = np.zeros([m, n])                                                       # Previous state (float64)
    u_curr = np.zeros([m, n])                                                       # Current state (float64)
    Gamma  = Gamma * nu * dt                                                        # Gammas with scientific precision (float64)

    # Indices for internal nodes
    core_slice  = (slice(1, m-1), slice(1, n-1))                                    # [1:-1, 1:-1] - internal nodes
    gamma_slice = (slice(1, m-1), slice(1, n-1), slice(None))                       # [1:-1, 1:-1, :] - Gamma weights
    
    # Indices for stencil neighbors
    neighbor_indices = [
        (slice(2, m),     slice(1, n-1)),                                           # E:  [2:,   1:-1] - East
        (slice(2, m),     slice(2, n)),                                             # NE: [2:,   2:]   - Northeast  
        (slice(1, m-1),   slice(2, n)),                                             # N:  [1:-1, 2:]   - North
        (slice(0, m-2),   slice(2, n)),                                             # NW: [0:-2, 2:]   - Northwest
        (slice(0, m-2),   slice(1, n-1)),                                           # W:  [0:-2, 1:-1] - West
        (slice(0, m-2),   slice(0, n-2)),                                           # SW: [0:-2, 0:-2] - Southwest
        (slice(1, m-1),   slice(0, n-2)),                                           # S:  [1:-1, 0:-2] - South
        (slice(2, m),     slice(0, n-2))                                            # SE: [2:,   0:-2] - Southeast
    ]
    
    # Indices for boundary conditions (pre-calculated)
    boundary_indices = {
        'inlet':      (slice(None), 0),                                             # [:,   0]   - Inlet (Dirichlet)
        'outlet':     (slice(None), n-1),                                           # [:,   n-1] - Outlet (Neumann)
        'outlet_ref': (slice(None), n-2),                                           # [:,   n-2] - Reference for Neumann
        'left':       (m-1,         slice(None)),                                   # [m-1, :]   - Left (Neumann)
        'left_ref':   (m-2,         slice(None)),                                   # [m-2, :]   - Reference for Neumann
        'right':      (0,           slice(None)),                                   # [0,   :]   - Right (Neumann)
        'right_ref':  (1,           slice(None))                                    # [1,   :]   - Reference for Neumann
    }

    # Initial condition
    u_prev[:, :] = 0                                                                # Initial Condition

    # Create 3D array with all neighbor values for einsum operation
    neighbor_arrays = np.zeros((m-2, n-2, 9))                                       # (internal_nodes_x, internal_nodes_y, 9_neighbors)
    
    # Solve the differential equation (loop with advanced vectorization)
    for k in range(1, t):                                                           # Iterate over time steps
        neighbor_arrays[:, :, 0] = u_prev[core_slice]                               # Center (current node)
        neighbor_arrays[:, :, 1] = u_prev[neighbor_indices[0]]                      # East
        neighbor_arrays[:, :, 2] = u_prev[neighbor_indices[1]]                      # Northeast
        neighbor_arrays[:, :, 3] = u_prev[neighbor_indices[2]]                      # North
        neighbor_arrays[:, :, 4] = u_prev[neighbor_indices[3]]                      # Northwest
        neighbor_arrays[:, :, 5] = u_prev[neighbor_indices[4]]                      # West
        neighbor_arrays[:, :, 6] = u_prev[neighbor_indices[5]]                      # Southwest
        neighbor_arrays[:, :, 7] = u_prev[neighbor_indices[6]]                      # South
        neighbor_arrays[:, :, 8] = u_prev[neighbor_indices[7]]                      # Southeast
        
        # EINSUM OPERATION: Vectorized multiplication and sum in a single line
        # 'ijk,ijk->ij' means: for each (i,j), sum over k of Gamma[i,j,k] * neighbor[i,j,k]
        u_curr[core_slice] = u_prev[core_slice] + np.einsum('ijk,ijk->ij', Gamma[gamma_slice], neighbor_arrays)

        # Boundary conditions using pre-calculated indices
        u_curr[boundary_indices['inlet']] = u_init                                  # Dirichlet at inlet
        u_curr[boundary_indices['outlet']] = u_curr[boundary_indices['outlet_ref']] # Neumann at outlet
        u_curr[boundary_indices['left']] = u_curr[boundary_indices['left_ref']]     # Neumann at left boundary
        u_curr[boundary_indices['right']] = u_curr[boundary_indices['right_ref']]   # Neumann at right boundary
        
        # Swap matrices for the next step
        u_prev, u_curr = u_curr, u_prev                                             # Optimized swap
    
    return u_prev                                                                   # u_prev contains the final solution

def Gammas(x, y, L):
    '''
    Calculates the Gamma coefficients for the Generalized Finite Differences (GFD) method.
    
    This function implements the vectorized calculation of Gamma coefficients that are
    fundamental for the GFD method. These coefficients allow approximating differential
    operators on unstructured meshes with high precision.
    
    Algorithm:
    1. For each internal node, coordinate differences with its 8 neighbors are calculated
    2. A matrix M is constructed with these differences according to the GFD scheme
    3. The pseudoinverse of M is calculated for each node
    4. It is multiplied by the differential operator L to obtain the Gamma coefficients
    
    Args:
        x (numpy.ndarray): X-coordinates of the mesh, matrix of shape (m,n)
        y (numpy.ndarray): Y-coordinates of the mesh, matrix of shape (m,n)
        L (numpy.ndarray): Differential operator, usually the Laplacian
        
    Returns:
        numpy.ndarray: Matrix of Gamma coefficients of shape (m,n,9), where the last
                      index corresponds to the central node (0) and its 8 neighbors (1-8)
    '''
    
    m, n = x.shape                                                                  # Mesh size
    Gamma = np.zeros([m, n, 9])                                                     # Gamma array with scientific precision
    
    # Create masks for internal nodes
    i_range = slice(1, m-1)                                                         # Create a mask for internal nodes in x
    j_range = slice(1, n-1)                                                         # Create a mask for internal nodes in y
    
    # Coordinates of central nodes
    x_center = x[i_range, j_range]                                                  # Central coordinates in x
    y_center = y[i_range, j_range]                                                  # Central coordinates in y
    
    # Calculate all dx and dy differences with scientific precision
    # Order of neighbors: E, NE, N, NW, W, SW, S, SE
    dx_neighbors = np.stack([
        x[2:m,   1:n-1] - x_center,                                                 # E:  (i+1, j)
        x[2:m,   2:n]   - x_center,                                                 # NE: (i+1, j+1)
        x[1:m-1, 2:n]   - x_center,                                                 # N:  (i,   j+1)
        x[0:m-2, 2:n]   - x_center,                                                 # NW: (i-1, j+1)
        x[0:m-2, 1:n-1] - x_center,                                                 # W:  (i-1, j)
        x[0:m-2, 0:n-2] - x_center,                                                 # SW: (i-1, j-1)
        x[1:m-1, 0:n-2] - x_center,                                                 # S:  (i,   j-1)
        x[2:m,   0:n-2] - x_center                                                  # SE: (i+1, j-1)
    ], axis=-1)
    
    dy_neighbors = np.stack([
        y[2:m,   1:n-1] - y_center,                                                 # E:  (i+1, j)
        y[2:m,   2:n]   - y_center,                                                 # NE: (i+1, j+1)
        y[1:m-1, 2:n]   - y_center,                                                 # N:  (i,   j+1)
        y[0:m-2, 2:n]   - y_center,                                                 # NW: (i-1, j+1)
        y[0:m-2, 1:n-1] - y_center,                                                 # W:  (i-1, j)
        y[0:m-2, 0:n-2] - y_center,                                                 # SW: (i-1, j-1)
        y[1:m-1, 0:n-2] - y_center,                                                 # S:  (i,   j-1)
        y[2:m,   0:n-2] - y_center                                                  # SE: (i+1, j-1)
    ], axis=-1)
    
    # Create M matrices for all internal nodes simultaneously
    M_matrices = np.stack([
        dx_neighbors,                                                               # First row: dx
        dy_neighbors,                                                               # Second row: dy
        dx_neighbors**2,                                                            # Third row: dx²
        dx_neighbors * dy_neighbors,                                                # Fourth row: dx*dy
        dy_neighbors**2                                                             # Fifth row: dy²
    ], axis=2)
    
    # Calculate pseudoinverse for all nodes using broadcasting
    # Reshape to apply pinv to each 5x8 matrix
    original_shape = M_matrices.shape[:2]                                           # (m-2, n-2)
    M_reshaped     = M_matrices.reshape(-1, 5, 8)                                   # (total_nodes, 5, 8)
    
    # Apply pseudoinverse to each matrix (maintaining scientific precision)
    M_pinv_list = []                                                                # List to store pseudoinverses
    for k in range(M_reshaped.shape[0]):                                            # Iterate over each 5x8 matrix
        pinv_result = np.linalg.pinv(M_reshaped[k])                                 # Pseudoinverse with full precision
        M_pinv_list.append(pinv_result)                                             # Add to the list
    
    M_pinv = np.array(M_pinv_list)                                                  # (total_nodes, 8, 5) float64
    M_pinv = M_pinv.reshape(original_shape[0], original_shape[1], 8, 5)             # (m-2, n-2, 8, 5)
    
    # Calculate YY = M_pinv @ L for all nodes
    L_flat = L.flatten()                                                            # Flatten operator L
    YY = np.einsum('ijkl,l->ijk', M_pinv, L_flat)                                   # (m-2, n-2, 8) with scientific precision
    
    # Calculate Gamma_0 = -sum(YY) for each node
    Gamma_0 = -np.sum(YY, axis=2)                                                   # (m-2, n-2) with scientific precision
    
    # Assign values to the Gamma matrix
    Gamma[1:m-1, 1:n-1, 0] = Gamma_0                                                # Gamma_0 (central node)
    Gamma[1:m-1, 1:n-1, 1:9] = YY                                                   # Gamma_1 to Gamma_8 (neighbors)
    
    return Gamma

def graficar(x, y, u_final, output_file):
    '''
    Generates a high-quality visualization of the simulation result.
    
    This function creates a graphical representation of the final solution of the diffusion
    equation, optimized to visualize the concentration in the skin mesh.
    
    Args:
        x (numpy.ndarray): X-coordinates of the mesh
        y (numpy.ndarray): Y-coordinates of the mesh
        u_final (numpy.ndarray): Final solution of the diffusion equation
        output_file (str): Path where the generated image will be saved
        
    Returns:
        None: The function saves the image in the specified file
    '''
    # Create figure with correct aspect ratio
    fig, ax = plt.subplots()
    
    # Configure levels for the contour (100 levels between 0 and 100)
    levels = np.linspace(0, 100, 100)
    plt.contourf(x, y, u_final, levels=levels, cmap='cool_r', vmin=0, vmax=100)
    
    # Configure axes without borders for a clean visualization
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    
    # Save image with high resolution and without margins
    plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0, facecolor='white')
    plt.close()


def main():
    '''
    Main function that executes the skin diffusion simulation.
    
    This function performs the following steps:
    1. Loads mesh data from a .mat file
    2. Configures simulation parameters
    3. Verifies numerical stability (CFL condition)
    4. Calculates GFD coefficients
    5. Executes the diffusion simulation
    6. Generates and saves a visualization of the result
    
    The function can be modified to change simulation parameters such as
    the diffusion coefficient (nu) and the initial concentration (u_init).
    '''
    
    # Load mesh data
    datos  = scipy.io.loadmat('region/skin41.mat')
    x, y   = datos["x"], datos["y"]                                                 # Mesh coordinates
    m, n  = x.shape                                                                 # Mesh dimensions
    t      = 3600                                                                   # Total simulation time (in seconds)
    T      = np.linspace(0, 3600, t)                                                # Time discretization
    dt     = T[1] - T[0]                                                            # Time step

    # Define problem parameters (these are what will vary in the problem)
    nu     = 1e-6                                                                   # Diffusion coefficient
    u_init = 24                                                                    # Concentration at the boundary

    # Verify numerical stability (CFL condition)
    dx_min = np.min(np.sqrt((x[1:, :] - x[:-1, :])**2 + (y[1:, :] - y[:-1, :])**2)) # Minimum dx size
    alpha = nu*dt/dx_min**2                                                         # Courant number
    while alpha > 0.5:                                                              # If the Courant number is too large
        print(f'Courant number is too large. C = {alpha}')
        t      = int(t * 1.1)
        T      = np.linspace(0, 3600, t)                                            # Time discretization
        dt     = T[1] - T[0]                                                        # Time step
        alpha = nu*dt/dx_min**2                                                     # Courant number

    # Calculate GFD coefficients
    L = np.vstack([[0], [0], [2], [0], [2]])                                        # Differential operator (Laplacian)
    Gamma = Gammas(x, y, L)                                                         # Calculate Gammas

    # Solve the diffusion equation using optimized Numba JIT
    u_final = difusion_skin_jit(m, n, t, dt, nu, u_init, Gamma)
    
    # Generate visualization
    graficar(x, y, u_final, 'skin.png')
    
    print(f"Simulation completed with {t} time steps")

if __name__ == "__main__":
    main()