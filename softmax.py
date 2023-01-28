from builtins import range
from ssl import SSL_ERROR_EOF
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)
    """
    loss = 0.0
    dW = np.zeros_like(W)
    for i in range(X.shape[0]):
      x = X[i,:]
      scores = []
      for j in range(W.shape[1]):
        score = 0.0
        for k in range(W.shape[0]):
          score += W[k, j]*x[k]
        scores.append(np.exp(score))

      scores = np.array(scores)/np.sum(scores)
      loss -= np.log(scores[y[i]])
      
      dW[:,y[i]] -= x
      dW += x.reshape(W.shape[0],1)*scores

    dW /= X.shape[0]
    loss /= X.shape[0]

    loss += np.sum(W**2)*reg
    dW += np.sum(W)*2*reg
    
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.
    """
    loss = 0.0
    dW = np.zeros_like(W)
    scores = np.exp(X.dot(W))
    scores_normalized = scores/scores.sum(axis=1).reshape(X.shape[0], 1)
    loss = -np.sum(np.log(scores_normalized[np.arange(X.shape[0]),y]))/X.shape[0]
    scores_normalized[np.arange(X.shape[0]),y]-=1
    dW = X.T.dot(scores_normalized)/X.shape[0]

    loss += np.sum(W**2)*reg
    dW += np.sum(W)*2*reg

    return loss, dW
