import torch




def model_loss_equation_ball(model, X_domain, nu):
    """
    Implement the part of the loss that makes sure the NN satisfies the equation on the domain

    Arguments :
        X = inputs form the dataset
        model = the model that we train
        nu = viscosity (float)
    Return :
        MSE on the pde
    """
    
    Y_pred = model(X_domain)

    u = Y_pred[:, 0]
    v = Y_pred[:, 1]
    p = Y_pred[:, 2]

    du = torch.autograd.grad(u, X_domain, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    dv = torch.autograd.grad(v, X_domain, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    dp = torch.autograd.grad(p, X_domain, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]

    u_x = du[:, 0]
    u_y = du[:, 1]
    v_x = dv[:, 0]
    v_y = dv[:, 1]
    p_x = dp[:, 0]
    p_y = dp[:, 1]

    u_xx = torch.autograd.grad(u_x, X_domain, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0][:, 0]
    u_yy = torch.autograd.grad(u_y, X_domain, grad_outputs=torch.ones_like(u_y), retain_graph=True, create_graph=True)[0][:, 1]
    v_xx = torch.autograd.grad(v_x, X_domain, grad_outputs=torch.ones_like(v_x), retain_graph=True, create_graph=True)[0][:, 0]
    v_yy = torch.autograd.grad(v_y, X_domain, grad_outputs=torch.ones_like(v_y), retain_graph=True, create_graph=True)[0][:, 1]

    #equations
    e1 = u*u_x + v*u_y + p_x - nu*(u_xx + u_yy)
    e2 = u*v_x + v*v_y + p_y - nu*(v_xx + v_yy)
    e3 = u_x + v_y  #zero divergence

    #compute the loss as the mean squared error of the equations
    loss = torch.mean(e1**2 + e2**2 + e3**2)
    return loss



def model_loss_boundary_ball(model, X_left, X_right, X_bot, X_top, X_ball, cond_left, cond_right, cond_bot, cond_top, cond_ball):


    UV_pred_bot = model(X_bot)[:, :2]
    UV_pred_top = model(X_top)[:, :2]
    UV_pred_left = model(X_left)[:, :2]
    UV_pred_right = model(X_right)[:, :2]
    UV_pred_ball = model(X_ball)[:, :2]

    DUDX_pred_right = torch.autograd.grad(UV_pred_right[:, 0], X_right, grad_outputs=torch.ones_like(UV_pred_right[:, 0]), retain_graph=True, create_graph=True, allow_unused = True)[0][:, 0]
    DVDX_pred_right = torch.autograd.grad(UV_pred_right[:, 1], X_right, grad_outputs=torch.ones_like(UV_pred_right[:, 0]), retain_graph=True, create_graph=True, allow_unused = True)[0][:, 0]

    # Reshaping pour s'assurer que ce sont des tensors 2D (batch_size, 1)
    DUDX_pred_right = DUDX_pred_right.view(-1, 1)
    DVDX_pred_right = DVDX_pred_right.view(-1, 1)

    # Concaténer les tensors le long de l'axe des colonnes (dim=1)
    DUDX_DVDX_pred_right = torch.concat([DUDX_pred_right, DVDX_pred_right], dim=1)

    e_left = UV_pred_left - cond_left
    e_right = DUDX_DVDX_pred_right - cond_right
    e_top = UV_pred_top - cond_top
    e_bot = UV_pred_bot - cond_bot
    e_ball = UV_pred_ball - cond_ball

    return torch.mean(e_left**2 + e_right**2 + e_bot**2 + e_top**2 + e_ball**2)