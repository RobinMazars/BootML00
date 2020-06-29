import numpy as np


def simple_predict(x, theta):
    """Computes the vector of prediction y_hat from two non-empty np.ndarray.
    Args:
    x: has to be an np.ndarray, a vector of dimension m * 1.
    theta: has to be an np.ndarray, a vector of dimension 2 * 1.
    Returns:
    y_hat as a np.ndarray, a vector of dimension m * 1.
    None if x or theta are empty np.ndarray.
    None if x or theta dimensions are not appropriate.
    Raises:
    This function should not raise any Exception.
    """
    if (not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray)
       or len(x) == 0 or len(theta) != 2):
        return None
    else:
        y = theta[0] + theta[1] * x
        return y.astype(float)


if __name__ == '__main__':
    x = np.arange(1, 6)
    # Example 1:,
    theta1 = np.array([-3, 1])
    y = simple_predict(x, theta1)
    print(y)
