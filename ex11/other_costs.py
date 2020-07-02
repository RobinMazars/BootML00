import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def mse_(y, y_hat):
    """
    Description:
    Calculate the MSE between the predicted output and the real output.
    Args:
    y: has to be a numpy.ndarray, a vector of dimension m * 1.
    y_hat: has to be a numpy.ndarray, a vector of dimension m * 1.
    Returns:
    mse: has to be a float.
    None if there is a matching dimension problem.
    Raises:
    This function should not raise any Exceptions.
    """
    if (not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray)
            or y.shape != y_hat.shape):
        return None
    else:
        return ((1 / y.size) * pow((y_hat - y), 2)).sum()


def rmse_(y, y_hat):
    """
    Description:
    Calculate the RMSE between the predicted output and the real output.
    Args:
    y: has to be a numpy.ndarray, a vector of dimension m * 1.
    y_hat: has to be a numpy.ndarray, a vector of dimension m * 1.
    Returns:
    rmse: has to be a float.
    None if there is a matching dimension problem.
    Raises:
    This function should not raise any Exceptions.
    """
    if (not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray)
            or y.shape != y_hat.shape):
        return None
    else:
        return np.sqrt(mse_(y, y_hat))


def mae_(y, y_hat):
    """
    Description:
    Calculate the MAE between the predicted output and the real output.
    Args:
    y: has to be a numpy.ndarray, a vector of dimension m * 1.
    y_hat: has to be a numpy.ndarray, a vector of dimension m * 1.
    Returns:
    mae: has to be a float.
    None if there is a matching dimension problem.
    Raises:
    This function should not raise any Exceptions.
    """
    if (not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray)
            or y.shape != y_hat.shape):
        return None
    else:
        return ((1 / y.size) * abs(y_hat - y)).sum()


def r2score_(y, y_hat):
    """
    Description:
    Calculate the R2score between the predicted output and the output.
    Args:
    y: has to be a numpy.ndarray, a vector of dimension m * 1.
    y_hat: has to be a numpy.ndarray, a vector of dimension m * 1.
    Returns:
    r2score: has to be a float.
    None if there is a matching dimension problem.
    Raises:
    This function should not raise any Exceptions.
    """
    if (not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray)
            or y.shape != y_hat.shape):
        return None
    else:
        return 1 - ((pow((y - y_hat), 2).sum()) / (pow((y - y.mean()), 2).sum()))


if __name__ == '__main__':
    x = np.array([0, 15, -9, 7, 12, 3, -21])
    y = np.array([2, 14, -13, 5, 12, 4, -19])
    # Mean squared error
    # your implementation
    print(mse_(x, y))
    # sklearn implementation
    print(mean_squared_error(x, y))
    # Output:
    # Root mean squared error
    # your implementation
    print(rmse_(x, y))
    # sklearn implementation not available: take the square root of MSE
    print(sqrt(mean_squared_error(x, y)))
    # Mean absolute error
    # your implementation
    print(mae_(x, y))

    # sklearn implementation
    print(mean_absolute_error(x, y))
    # R2-score
    # your implementation
    print(r2score_(x, y))
    # sklearn implementation
    print(r2_score(x, y))
