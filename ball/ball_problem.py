import torch
import torch.nn as nn
import numpy as np
from matplotlib.path import Path

def boundary_conditions_ball_problem(n_ball_sample, n_border_sample_1D, center_ball = [0, 0], ray = 1) : 

    x_c = center_ball[0]
    y_c = center_ball[1]
    r_c = ray

    start_x = -2
    end_x = 5
    start_y = -5
    end_y = 5

    x_dom_bound = torch.linspace(start_x, end_x, n_border_sample_1D)
    y_dom_bound = torch.linspace(start_y, end_y, n_border_sample_1D)

    X_left = torch.zeros((n_border_sample_1D, 2))
    X_left[:, 0] = start_x
    X_left[:, 1] = y_dom_bound

    X_left.requires_grad = True

    UV_left = torch.zeros((n_border_sample_1D, 2))  ## On met que 2 et pas 3 car on n'impose rien sur la pression
    UV_left[:, 0] = 1 
    UV_left[:, 1] = 0

    ## Outlet side of the box

    X_right = torch.zeros((n_border_sample_1D, 2))
    X_right[:, 0] = end_x
    X_right[:, 1] = y_dom_bound

    X_right.requires_grad = True

    ## Outflow condition on the outlet 

    DUDX_DVDX_right = torch.zeros((n_border_sample_1D, 2))  ## On met 2 ici car pareil on va imposer une condition sur U et V mais pas sur p
    DUDX_DVDX_right[:, 0] = 0
    DUDX_DVDX_right[:, 1] = 0

    ## Top

    X_top = torch.zeros((n_border_sample_1D, 2))
    X_top[:, 0] = x_dom_bound
    X_top[:, 1] = end_y

    X_top.requires_grad = True

    ## No slip condition on the top

    UV_top = torch.zeros((n_border_sample_1D, 2))  ## On met que 2 et pas 3 car on n'impose rien sur la pression
    UV_top[:, 0] = 1
    UV_top[:, 1] = 0

    ## Bottom

    X_bottom = torch.zeros(((n_border_sample_1D, 2)))
    X_bottom[:, 0] = x_dom_bound
    X_bottom[:, 1] = start_y

    X_bottom.requires_grad = True

    ## No slip condition on the bottom

    UV_bottom = torch.zeros((n_border_sample_1D, 2))  ## On met que 2 et pas 3 car on n'impose rien sur la pression
    UV_bottom[:, 0] = 1
    UV_bottom[:, 1] = 0

    ## Now the ball


    angles = torch.linspace(0, 2 * torch.pi, steps=n_ball_sample)
    X_ball = torch.zeros((n_ball_sample, 2))
    X_ball[:, 0] = x_c + r_c * torch.cos(angles)
    X_ball[:, 1] = y_c + r_c * torch.sin(angles)

    X_ball.requires_grad = True

    ## Condition on the ball : no slip

    UV_BALL = torch.zeros((n_ball_sample, 2))
    UV_BALL[:, 0] = 0
    UV_BALL[:, 1] = 0

    return X_left, X_right, X_top, X_bottom, X_ball, UV_left, DUDX_DVDX_right, UV_top, UV_bottom, UV_BALL


def generate_points_between_curves(X1, X2, n_points):
    ## X2 = Points de la courbe extérieur 
    ## X1 = Points de la courbe intérieur 

    # Étape 1 : Définir la bounding box
    x_min = min(np.min(X1[:, 0]), np.min(X2[:, 0]))
    x_max = max(np.max(X1[:, 0]), np.max(X2[:, 0]))
    y_min = min(np.min(X1[:, 1]), np.min(X2[:, 1]))
    y_max = max(np.max(X1[:, 1]), np.max(X2[:, 1]))
    
    # Étape 2 : Créer les objets Path pour les courbes
    path_X1 = Path(X1)  # Courbe intérieure
    path_X2 = Path(X2)  # Courbe extérieure

    # Étape 3 : Générer des points aléatoires et filtrer
    points = []
    while len(points) < n_points:
        # Générer des points aléatoires dans la bounding box
        x_random = np.random.uniform(x_min, x_max, size=n_points)
        y_random = np.random.uniform(y_min, y_max, size=n_points)
        candidates = np.vstack((x_random, y_random)).T
        
        # Filtrer les points : intérieur X2 mais extérieur X1
        for point in candidates:
            if path_X2.contains_point(point) and not path_X1.contains_point(point):
                points.append(point)
                if len(points) >= n_points:
                    break

    return np.array(points)

def get_domain(X_left, X_right, X_top, X_bottom, X_ball) : 
    X_2 = torch.concat([X_bottom, X_right, torch.flip(X_top, dims=[0]), torch.flip(X_left, dims = [0])])
    X_2 = X_2.detach().numpy()
    X_1 = X_ball.detach().numpy()
    X_t = generate_points_between_curves(X_1, X_2, 2500)
    X_t = torch.from_numpy(X_t)
    X_t = X_t.float()
    X_t.requires_grad = True
    return X_t










