import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  N = X.shape[0]
  C = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(N):
    f = X[i].dot(W)
    f -= np.max(f) # numeric stability
    ef = np.exp(f)
    sef = np.sum(ef)
    loss += -f[y[i]] + np.log(sef)
    dW[:,y[i]] -= X[i]
    for j in range(C):
      dW[:,j] += ef[j]/sef * X[i]
    
  loss /= N
  dW /= N
  # regularization
  loss += reg * np.sum(W*W)
  dW += reg * 2 * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  N = X.shape[0]
  C = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  F = X @ W # (N,C) matrix
  F -= np.max(F, axis=1, keepdims=True) # remove (N,1) column wise
  # keepdims makes shape be (N,1) instead of (1,N)  
  F = np.exp(F)
  F = np.divide(F, np.sum(F, axis=1, keepdims=True))
  # Fij = F[i][j] / sum_j F[i][j]
  
  loss -= np.log(F[np.arange(N), y]).sum()
  
  F[np.arange(N),y] -= 1 # -f_y[i]
  dW += X.T @ F
  
  loss /= N
  dW /= N
  
  # regularization
  loss += 2 * reg * np.sum(W*W)
  dW += reg * 2 * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
