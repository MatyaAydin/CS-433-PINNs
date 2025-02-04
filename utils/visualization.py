
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.interpolate import griddata




def visualize_unsteady_time(model, time_index, index_quantity, X, Y, mean_x, std_x, mean_y, std_y, nb_nodes) :
    
    X_test = X[time_index*nb_nodes:(time_index+1)*nb_nodes]
    Y_test = Y[time_index*nb_nodes:(time_index+1)*nb_nodes]
    X_test_standardized = (X_test - mean_x)/ std_x
    Y_test_pred = model(X_test_standardized)
    Y_test_pred = Y_test_pred * std_y + mean_y

    x_test = X_test[:,0]
    y_test = X_test[:,1]
    t = 0.01*(time_index+1)

    prediction = Y_test_pred[:,index_quantity]
    simulation = Y_test[:,index_quantity]
    
    z1 = prediction.detach().numpy()
    z2 = simulation.detach().numpy()

    z_min = min(min(z1), min(z2))
    z_max = max(max(z1), max(z2))

    norm = Normalize(vmin=z_min, vmax=z_max)

    if index_quantity == 0 : name = "P"
    if index_quantity == 1 : name = "U"
    if index_quantity == 2 : name = "V"

    plt.figure(figsize=(11, 7))
    plt.scatter(x_test.detach().numpy(), y_test.detach().numpy(), c=prediction.detach().numpy(), cmap='jet', s=5, norm=norm)
    cbar1 = plt.colorbar()
    cbar1.set_label(f'{name}')
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Prediction (t = "+str(np.round(t, 2))+"s)")
    
    plt.figure(figsize=(11, 7))
    plt.scatter(x_test.detach().numpy(), y_test.detach().numpy(), c=simulation.detach().numpy(), cmap='jet', s=5, norm=norm)
    cbar1 = plt.colorbar()
    cbar1.set_label(f'{name}')
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Simulation (t = "+str(np.round(t, 2))+"s)")
    


def vis_velocity(X_flat, values, name, grid_res=100, method='linear'):
    """
    2-D visualization of the x-velocity.

    Args:
        X_flat:  (Nx2) array flattened meshgrid.
        values: (N,) array of the x-velocity values.
        grid_res int: grid resolution.
        method: str, interpolation method.
        name: str, name of the plot.
    """
    #convert tensor in array if tensors are given
    X_flat = np.array(X_flat)
    values = np.array(values)

    #domain limit
    x_min, x_max = np.min(X_flat[:, 0]), np.max(X_flat[:, 0])
    y_min, y_max = np.min(X_flat[:, 1]), np.max(X_flat[:, 1])

    #create grid
    x_grid = np.linspace(x_min, x_max, grid_res)
    y_grid = np.linspace(y_min, y_max, grid_res)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

    #interpolation on grid
    Z_grid = griddata(X_flat, values, (X_grid, Y_grid), method=method)

    # Visualisation
    plt.figure(figsize=(8, 6))
    plt.contourf(X_grid, Y_grid, Z_grid, levels=100, cmap='viridis')
    plt.title("Velocity Field")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.savefig(f'./plot/{name}.pdf')
    plt.show()



def vis_pressure(x, y, p, name):
    """
    2-D visualization of the pressure field.

    Args:
        x: (Nx1) array of x-coordinates.
        y: (Nx1) array of y-coordinates.
        p: (NxN) array of pressure values.
        name: str, name of the plot.
    """
    cp = plt.contourf(x, y, p, levels=50, cmap='coolwarm')
    cbar_p = plt.colorbar(cp)
    cbar_p.set_label('Pressure')

    # Set labels, title, and axis properties
    plt.title('Pressure Field')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.savefig(f'./plot/{name}.pdf')
    plt.show()


def visualize_sol(x, y, u, v, p, name):

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Compute the velocity magnitude for color coding
    velocity_magnitude = np.sqrt(u**2 + v**2)

    # Velocity field with color-coded magnitude
    vel_plot = ax[0].quiver(x, y, u, v, velocity_magnitude, cmap='viridis', angles='xy', scale_units='xy', scale=1)
    cbar_vel = fig.colorbar(vel_plot, ax=ax[0])
    cbar_vel.set_label('Velocity Magnitude')
    ax[0].set_title('Velocity Field (Quiver Plot with Magnitude)')
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    ax[0].axis('equal')

    # Pressure field as a contour plot
    cp = ax[1].contourf(x, y, p, levels=50, cmap='coolwarm')
    cbar_p = fig.colorbar(cp, ax=ax[1])
    cbar_p.set_label('Pressure')
    ax[1].set_title('Pressure Field')
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('y')
    ax[1].axis('equal')

    # Save and show the plot
    plt.tight_layout()
    plt.savefig(f'./plot/{name}.pdf')
    plt.show()



if __name__ == "__main__":
    nb_points = 51
    x = np.linspace(-0.5, 1, nb_points)
    y = np.linspace(-0.5, 1.5, nb_points)
    X, Y = np.meshgrid(x, y)

    X_flat = np.array([X.flatten(), Y.flatten()]).T
    print(X_flat.shape)

    
    #parameters
    Re = 40 
    nu = 1/Re 
    lambda_ = -Re/2 + np.sqrt((Re/2)**2 + 4 *np.pi**2)


    #analytical solution
    U = 1 - np.exp(lambda_*X_flat[:,0])*np.cos(2*np.pi*X_flat[:,1])
    V = (lambda_/(2*np.pi))*np.exp(lambda_*X_flat[:,0])*np.sin(2*np.pi*X_flat[:,1])
    P = 0.5*(1 - np.exp(2*lambda_*X_flat[:,0]))

    #change path if you wish to generate the analytical solution
    vis_velocity(X_flat, U, 'vel_true', grid_res=200, method='cubic')
    vis_pressure(X, Y, P.reshape(X.shape), 'pres_true')