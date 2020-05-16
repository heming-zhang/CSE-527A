import numpy as np

def grdescent(func, w0, stepsize, maxiter, tolerance = 1e-02):
    # INPUT:
    # func function to minimize
    # w_trained = initial weight vector
    # stepsize = initial gradient descent stepsize
    # tolerance = if norm(gradient)<tolerance, it quits
    #
    # OUTPUTS:
    # w = final weight vector

    eps = 2.2204e-14 #minimum step size for gradient descent

    # YOUR CODE HERE
    i = 0
    w = w0
    loss, gradient = func(w0)
    # stop early if the norm of the gradient is less than the tolerance value.
    # When the norm of the gradient is tiny it means that you have arrived at a minimum
    while np.linalg.norm(gradient) >= tolerance and i < maxiter:
        # update w from the
        w = w - stepsize * gradient
        new_loss, gradient = func(w)
        # increase the stepsize by a factor of 1.01 each iteration where the loss goes down,
        # and decrease it by a factor 0.5 if the loss went up
        if new_loss < loss:
            stepsize *= 1.01
        elif new_loss >= loss:
            w = w + stepsize * gradient
            new_stepsize = stepsize * 0.5
            if new_stepsize <= eps:
                stepsize = eps
            else:
                stepsize = new_stepsize
        loss = new_loss
        i += 1
    return w
