import os
import sr_decoder
import sr_encoder
import pywt
import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt


def main():
	# Read image
	img_src = "lena.jpg"
	dirname = os.path.dirname(__file__)
	filename = os.path.join(dirname, img_src)
	image = img.imread(img_src)

	# pywt.wavedec2(data, wavelet, mode='symmetric', level=None, axes=(-2, -1))
	# http://pywavelets.readthedocs.io/en/latest/ref/2d-dwt-and-idwt.html?highlight=haar#d-multilevel-decomposition-using-wavedec2
	coeffs = iwt2(image)#pywt.wavedec2(image, 'db1')
	print(coeffs)
	# Show the image
	plt.imshow(iiwt2(coeffs))#pywt.waverec2(coeffs, 'db1'))
	plt.gray()
	plt.show()
	

def iwt(array):
    output = np.zeros_like(array)
    nx, ny = array.shape
    x = nx // 2
    for j in range(ny):
        output[0:x,j] = (array[0::2,j] + array[1::2,j])//2
        output[x:nx,j] = array[0::2,j] - array[1::2,j]
    return output

def iiwt(array):
    output = np.zeros_like(array)
    nx, ny = array.shape
    x = nx // 2
    for j in range(ny):
        output[0::2,j] = array[0:x,j] + (array[x:nx,j] + 1)//2
        output[1::2,j] = output[0::2,j] - array[x:nx,j]
    return output

def iwt2(array):
    return iwt(iwt(array.astype(int)).T).T

def iiwt2(array):
    return iiwt(iiwt(array.astype(int).T).T)


if __name__ == "__main__":
	main()


