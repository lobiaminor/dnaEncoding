import io
import sys
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
	coeffs = pywt.wavedec2(image, 'db1')
	print(coeffs)
	# Show the image
	plt.imshow(pywt.waverec2(coeffs, 'db1'))
	plt.gray()
	plt.show()
	

if __name__ == "__main__":
    main()