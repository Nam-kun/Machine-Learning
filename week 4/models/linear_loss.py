import numpy as np

def linear_loss_naive(W, X, y, reg):
    """
    Linear loss function, naive implementation (with loops)

    Inputs have dimension D, there are N examples.

    Inputs:
    - W: A numpy array of shape (D, 1) containing weights.
    - X: A numpy array of shape (N, D) containing data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where c is a real number.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    lo = 0.0
    dW = np.zeros_like(W)
    for i in range(len(W)):
        dW[i] = 0
        for j in range(len(y)):
            #dW[i] += ((X[j][i]**2)*W[i] - X[j][i]*y[j])
            dW[i] += X[j][i]
        dW[i] = dW[i]**2
        dW[i] = dW[i]/(2*len(y))
    XW = W*X
    for i in range(len(y)):
        for j in range(len(W)):
            lo += (W[j]*X[i][j] - y[i])
        loss += lo**2
    loss = loss/(2*len(y))
    #############################################################################
    # TODO: Compute the linear loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def linear_loss_vectorized(W, X, y, reg):
    """
    Linear loss function, vectorized version.

    Inputs and outputs are the same as linear_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    for i in range(len(W)):
        dW[i] = 0
        for j in range(len(y)):
            #dW[i] += ((X[j][i]**2)*W[i] - X[j][i]*y[j])
            dW[i] += X[j][i]
        dW[i] = dW[i]**2
        dW[i] = dW[i]/(2*len(y))
    XW = W*X
    for i in range(len(y)):
        lo = (XW[i] - y[i])
        loss += lo**2
    loss = loss/(2*len(y))

    #############################################################################
    # TODO: Compute the linear loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    pass
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW