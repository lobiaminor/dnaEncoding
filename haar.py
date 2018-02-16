import numpy as np


def iwt(array):
    """ 1D Haar analysis of the input signal."""
    output = np.zeros_like(array)
    nx, ny = array.shape
    x = nx // 2
    for j in range(ny):
        output[0:x,j] = (array[0::2,j] + array[1::2,j])//2
        output[x:nx,j] = array[0::2,j] - array[1::2,j]
    return output


def iiwt(array):
    """ 1D Haar synthesis of the input signal."""
    output = np.zeros_like(array)
    nx, ny = array.shape
    x = nx // 2
    for j in range(ny):
        output[0::2,j] = array[0:x,j] + (array[x:nx,j] + 1)//2
        output[1::2,j] = output[0::2,j] - array[x:nx,j]
    return output


def iwt2(array):
    """ 2D Haar analysis of the input image.
    Params:
    - array: numpy matrix representing the image
    """
    return iwt(iwt(array.astype(int)).T).T


def iiwt2(array):
    """ 2D Haar synthesis of the input (transformed) image.
    Params:
    - array: numpy matrix representing the image
    """
    return iiwt(iiwt(array.astype(int).T).T)


def iwtn(image, n):
    """ Performs the n-levels Haar wavelet transform of the given image.
    Params:
    - image: numpy matrix representing the image
    - n: number of decomposition levels
    """
    x_axis, y_axis = image.shape
    new_image = np.zeros(image.shape, dtype=np.int64)
    new_image[:] = image
    for i in range(n):
        midx = int(x_axis / (np.power(2, i)))
        midy = int(y_axis / (np.power(2, i)))

        LL = np.zeros((midx, midy), dtype=np.int64)
        LL[:] = new_image[0:midx, 0:midy]

        new_image[0:midx, 0:midy] = iwt2(LL)

    return new_image


def iiwtn(coeffs, n):
    """ Performs the n-levels inverse Haar wavelet transform of the
    given transformed image.
    Params:
    - coeffs: numpy matrix representing the transformed image
    - n: number of decomposition levels
    """
    x_axis, y_axis = coeffs.shape
    new_image = np.zeros(coeffs.shape, dtype=np.int64)
    new_image[:] = coeffs
    for i in reversed(range(n)):
        midx = int(x_axis / (np.power(2, i)))
        midy = int(y_axis / (np.power(2, i)))

        LL = np.zeros((midx, midy), dtype=np.int64)
        LL[:] = new_image[0:midx, 0:midy]

        new_image[0:midx, 0:midy] = iiwt2(LL)

    return new_image
