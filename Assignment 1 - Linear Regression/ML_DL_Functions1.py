import numpy as np

def LeastSquares(X,y):
  '''
    Calculates the Least squares solution to the problem X*theta=y using the least squares method
    :param X: numpy input matrix, size [N,m+1] (feature 0 is a column of 1 for bias)
    :param y: numpy input vector, size [N]
    :return theta = (Xt*X)^(-1) * Xt * y: numpy output vector, size [m+1]
    N is the number of samples and m is the number of features=28
  '''
  Xt = np.transpose(X)
  theta = np.dot(np.linalg.pinv(np.dot(Xt, X)), np.dot(Xt, y))

  return theta

def classification_accuracy(model,X,s):
  '''
    calculate the accuracy for the classification problem
    :param model: the classification model class
    :param X: numpy input matrix, size [N,m]
    :param s: numpy input vector of ground truth labels, size [N]
    :return: accuracy of the model = (correct classifications)/(total classifications) type float
    N is the number of samples and m is the number of features=28
  '''
  predictions = model.predict(X)
  correct = sum(predictions == s)
  overall = len(s)
  return correct / overall

def linear_regression_coeff_submission():
  '''
    copy the values from your notebook into a list in here. make sure the values
    seperated by commas
    :return: list of coefficiants for the linear regression problem. length 28
  '''
  return  [
    -0.04939048, 0.02420877, -0.02521159, 0.00721053, -0.02021114, -0.01926598,
    0.07089107, 0.01178233, 0.00295448, -0.05615385, 0.06644596, 0.01104797,
    0.08540356, 0.13905259, 0.78071972, 0.03204459, 0.03897653, 0.0042018,
    0.03501683, -0.00142136, 0.03204224, 0.02028198, 0.02153471, -0.03757443,
    -0.02711017, 0.02757323, -0.03063815, -0.02349492
]

def linear_regression_intrcpt_submission():
  '''
    copy the intercept value from your notebook into here.
    :return: the intercept value. type float
  '''
  return -1.82411282145214e-16

def classification_coeff_submission():
  '''
    copy the values from your notebook into a list in here. make sure the values
    seperated by commas
    :return: list of list of coefficiants for the classification problem.  length 28
  '''
  return [
    [
        -3.11760443e-01, -1.47421714e-01,  2.72624003e-01, -1.63809827e-01,
        -2.04286747e-01, -9.63256295e-02, -1.61750476e-02, -1.34510995e-01,
         2.08664482e-01, -2.24768241e-04, -4.90470831e-01, -1.79819316e-02,
        -3.24492343e-02,  8.83970088e-01,  2.95168152e+00, -3.55140660e-01,
        -1.93177822e-03, -1.29889296e-01, -2.92323359e-02, -9.10285410e-02,
        -4.55586116e-02,  4.38063997e-02,  6.81730717e-03, -3.13005582e-01,
        -2.62339599e-01,  1.33747099e-01,  1.51219200e-01,  2.48445649e-01
    ]
]

def classification_intrcpt_submission():
  '''
    copy the intercept value from your notebook into here.
    :return: list with the intercept value. length 1
  '''
  return [0.44279433]


def classification_classes_submission():
  '''
    copy the classes values from your notebook into a list in here. make sure the values
    seperated by commas
    :return: list of classes for the classification problem. length 2.
  '''
  return [0, 1]